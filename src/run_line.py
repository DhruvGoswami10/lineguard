"""The line runner: the whole online workflow, end to end.

    python -m src.run_line --source samples/line_demo.mp4            # the recorded line (demo)
    python -m src.run_line --source 0 --display                       # a webcam, live window
    python -m src.run_line --source samples/ --model ae               # replay a folder, one part per image

Per frame:  acquisition (find the part in the station) -> trigger (collect N frames)
Per part:   vote (median score) -> decision (PASS / REVIEW / REJECT + severity)
            -> QC log (SQLite) -> reject signal -> alerts -> saved crop + heatmap
            -> annotated frame (written to a video, optionally shown live)

The operator dashboard (`streamlit run src/dashboard.py`) reads the same QC log while
this runs, and the thresholds it writes to the settings table are picked up here for
the next part: no restart, no retraining.

Outputs (default folder results/line/):
    qc_log.sqlite               every decision (src/qclog.py)
    reject_events.jsonl         the reject signals (src/reject.py)
    parts/<part>.png            the centred crop the decision was made on
    parts/<part>_heatmap.png    input | heatmap | verdict figure
    <source>_annotated.mp4      the camera feed with the station, verdicts and KPIs drawn on
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.acquisition import ROI, FolderSource, PresenceTrigger, crop_part, find_parts, open_source
from src.alerts import AlertMonitor
from src.common import get_device, set_seed
from src.decision import Rule, decide, rule_from_checkpoint, vote
from src.inference import MODEL_KINDS, inspect, load_model
from src.make_line_video import STATION
from src.qclog import QCLog
from src.reject import RejectSignal
from src.visualize import MODEL_NAMES, overlay, save_verdict_figure

PANEL_HEIGHT = 200
COLOURS = {"PASS": (80, 180, 60), "REVIEW": (0, 165, 255), "REJECT": (40, 40, 230),
           "empty": (160, 160, 160), "collecting": (0, 200, 255)}  # BGR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LineGuard on a video, a camera or a folder.")
    p.add_argument("--source", required=True, help="video file, camera index (0, 1, ...) or image folder")
    p.add_argument("--model", default="foundad", choices=list(MODEL_KINDS))
    p.add_argument("--weights", default="weights")
    p.add_argument("--roi", default=",".join(str(v) for v in STATION),
                   help="inspection station x,y,w,h in frame pixels")
    p.add_argument("--votes", type=int, default=5, help="frames scored per part (median wins)")
    p.add_argument("--dark", type=int, default=160, help="grey level below which pixels count as the part")
    p.add_argument("--reverse", action="store_true", help="parts travel right to left (frame is mirrored)")
    p.add_argument("--out", default="results/line", help="output folder (log, crops, video)")
    p.add_argument("--db", default=None, help="QC log file (default: <out>/qc_log.sqlite)")
    p.add_argument("--reset", action="store_true", help="start a fresh QC log and event log")
    p.add_argument("--no-video", action="store_true", help="do not write the annotated video")
    p.add_argument("--display", action="store_true", help="show the annotated frames in a window (q quits)")
    p.add_argument("--max-parts", type=int, default=0, help="stop after this many parts (0 = all)")
    p.add_argument("--device", default="cpu", help="cpu (default), auto or cuda")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def model_version(path: Path) -> str:
    return path.stem + ":" + hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def current_rule(log: QCLog, default: Rule) -> Rule:
    """The rule to decide with: the operator's settings if any, else the checkpoint's."""
    s = log.get_settings()
    return Rule(t_low=s.get("t_low", default.t_low), t_high=s.get("t_high", default.t_high),
                moderate_pct=s.get("moderate_pct", default.moderate_pct),
                severe_pct=s.get("severe_pct", default.severe_pct))


def ensure_settings(log: QCLog, rule: Rule, monitor: AlertMonitor) -> None:
    """Write the defaults once, so the dashboard has something to show and edit."""
    if not log.get_settings():
        log.set_settings(t_low=rule.t_low, t_high=rule.t_high, moderate_pct=rule.moderate_pct,
                         severe_pct=rule.severe_pct, baseline_rate=monitor.baseline_rate,
                         window=monitor.window, stall_seconds=monitor.stall_seconds)


def apply_alert_settings(log: QCLog, monitor: AlertMonitor) -> None:
    s = log.get_settings()
    monitor.baseline_rate = s.get("baseline_rate", monitor.baseline_rate)
    monitor.window = int(s.get("window", monitor.window))
    monitor.stall_seconds = s.get("stall_seconds", monitor.stall_seconds)


class Line:
    """Holds the subsystems and runs one part through all of them."""

    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        (self.out / "parts").mkdir(parents=True, exist_ok=True)
        db_path = Path(args.db) if args.db else self.out / "qc_log.sqlite"
        if args.reset:
            db_path.unlink(missing_ok=True)
            (self.out / "reject_events.jsonl").unlink(missing_ok=True)
        self.device = get_device(args.device)
        weights = Path(args.weights) / f"{args.model}.pt"
        self.model, self.ckpt = load_model(weights, self.device)
        self.version = model_version(weights)
        self.default_rule = rule_from_checkpoint(self.ckpt)
        self.log = QCLog(db_path)
        self.monitor = AlertMonitor()
        ensure_settings(self.log, self.default_rule, self.monitor)
        self.reject = RejectSignal(self.out / "reject_events.jsonl", beep=not args.quiet)
        self.run_tag = time.strftime("%H%M%S")
        self.n_parts = 0
        self.verdicts = []            # this run's verdicts, for the control chart
        self.last = None              # last decision summary for the display
        self.last_overlay = None      # last heatmap inset (BGR)
        self.alert_text = ""

    def score_frame(self, image):
        return inspect(self.model, self.ckpt, image, self.device, with_heatmap=True)

    def finish_part(self, frames: list[tuple], source_name: str) -> dict:
        """frames: [(crop PIL image, inspect result)]. Vote, decide, log, signal, save."""
        scores = [r["score"] for _, r in frames]
        final = vote(scores)
        image, result = min(frames, key=lambda fr: abs(fr[1]["score"] - final))  # the median frame
        rule = current_rule(self.log, self.default_rule)
        decision = decide(final, rule)
        self.n_parts += 1
        part_id = f"{self.run_tag}-{self.n_parts:04d}"

        crop_path = self.out / "parts" / f"{part_id}.png"
        heat_path = self.out / "parts" / f"{part_id}_heatmap.png"
        image.save(crop_path)
        shown = dict(result, score=final, threshold=rule.t_high, t_low=rule.t_low, verdict=decision.verdict)
        save_verdict_figure(image, shown, self.args.model, heat_path,
                            f"LineGuard | {MODEL_NAMES[self.args.model]} | {part_id}")

        row_id = self.log.record(part_id=part_id, model=self.args.model, model_version=self.version,
                                 score=final, t_low=rule.t_low, t_high=rule.t_high,
                                 verdict=decision.verdict, severity=decision.severity,
                                 frames=len(frames), source=source_name,
                                 image_path=str(crop_path), heatmap_path=str(heat_path))
        if decision.reject:
            self.reject.fire(part_id, final, decision.severity, row_id)
        self.verdicts.append(decision.verdict)
        apply_alert_settings(self.log, self.monitor)
        alert = self.monitor.check_rate(self.verdicts)
        if alert:
            self.log.add_alert(alert["kind"], alert["message"], alert["value"])
            self.alert_text = alert["message"]
            print(f"ALERT: {alert['message']}")
        self.last = {"part_id": part_id, "score": final, "verdict": decision.verdict,
                     "severity": decision.severity, "frames": len(frames), "rule": rule}
        self.last_overlay = cv2.cvtColor((overlay(image, result["heatmap"],
                                                  1.0 if self.args.model == "cnn" else rule.t_high,
                                                  result.get("map_floor", 0.0)) * 255).astype(np.uint8),
                                         cv2.COLOR_RGB2BGR)
        if not self.args.quiet:
            sev = f" ({decision.severity})" if decision.severity else ""
            print(f"{part_id}  score {final:.5g}  {decision.verdict}{sev}  "
                  f"[{len(frames)} frame{'s' if len(frames) > 1 else ''}, t_low {rule.t_low:.4g}, t_high {rule.t_high:.4g}]")
        return self.last

    def check_stall(self, seconds_since_last_part: float) -> None:
        alert = self.monitor.check_stall(seconds_since_last_part)
        if alert:
            self.log.add_alert(alert["kind"], alert["message"], alert["value"])
            self.alert_text = alert["message"]
            print(f"ALERT: {alert['message']}")

    # ---- drawing -------------------------------------------------------------
    def annotate(self, frame: np.ndarray, roi: ROI, state: str, part) -> np.ndarray:
        """The frame with the station box, the part box and a KPI panel underneath."""
        canvas = np.zeros((frame.shape[0] + PANEL_HEIGHT, frame.shape[1], 3), np.uint8)
        canvas[: frame.shape[0]] = frame
        colour = COLOURS.get(state, COLOURS["empty"])
        cv2.rectangle(canvas, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), colour, 3)
        cv2.putText(canvas, "inspection station", (roi.x + 8, roi.y + 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, colour, 2, cv2.LINE_AA)
        if part is not None:
            cv2.rectangle(canvas, (part.x, part.y), (part.x + part.w, part.y + part.h), colour, 2)

        y0 = frame.shape[0]
        canvas[y0:] = (30, 30, 30)
        counts = self.log.counts()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"LineGuard | {MODEL_NAMES[self.args.model]} | {self.args.source}",
                    (20, y0 + 32), font, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
        if self.last:
            c = COLOURS[self.last["verdict"]]
            sev = f"  {self.last['severity']}" if self.last["severity"] else ""
            cv2.putText(canvas, f"{self.last['verdict']}{sev}", (20, y0 + 90), font, 1.4, c, 3, cv2.LINE_AA)
            cv2.putText(canvas, f"{self.last['part_id']}   score {self.last['score']:.4g}   "
                        f"review >= {self.last['rule'].t_low:.4g}   reject >= {self.last['rule'].t_high:.4g}   "
                        f"{self.last['frames']} frames", (20, y0 + 125), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "waiting for the first part", (20, y0 + 90), font, 0.9, (160, 160, 160), 2,
                        cv2.LINE_AA)
        kpi = (f"inspected {counts['total']}   PASS {counts['PASS']}   REVIEW {counts['REVIEW']}   "
               f"REJECT {counts['REJECT']}   reject rate {counts['reject_rate']:.0%}")
        cv2.putText(canvas, kpi, (20, y0 + 165), font, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
        if self.alert_text:
            cv2.putText(canvas, "ALERT: " + self.alert_text[:70], (20, y0 + 190), font, 0.55, (0, 165, 255), 1,
                        cv2.LINE_AA)
        if self.last_overlay is not None:
            inset = cv2.resize(self.last_overlay, (PANEL_HEIGHT - 20, PANEL_HEIGHT - 20))
            x = frame.shape[1] - inset.shape[1] - 10
            canvas[y0 + 10:y0 + 10 + inset.shape[0], x:x + inset.shape[1]] = inset
            cv2.putText(canvas, "heatmap", (x, y0 + 8 + inset.shape[0] + 0), font, 0.45, (200, 200, 200), 1,
                        cv2.LINE_AA)
        return canvas


def run_folder(line: Line, source: FolderSource) -> None:
    """Replay mode: every image file is one captured, already centred part."""
    print(f"Replaying {len(source)} images from {line.args.source} ...")
    while True:
        ok, frame = source.read()
        if not ok:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        line.finish_part([(image, line.score_frame(image))], source.current_name)
        if line.args.max_parts and line.n_parts >= line.args.max_parts:
            break


def run_video(line: Line, source) -> None:
    """Video / camera mode: the trigger decides which frames to score."""
    args = line.args
    ok, frame = source.read()
    if not ok:
        sys.exit("Error: the source produced no frames.")
    roi = ROI.parse(args.roi).clip(frame.shape[1], frame.shape[0])
    trigger = PresenceTrigger(args.votes)
    writer = None
    if not args.no_video:
        name = ("camera" if source.is_camera else Path(args.source).stem) + "_annotated.mp4"
        writer = cv2.VideoWriter(str(line.out / name), cv2.VideoWriter_fourcc(*"mp4v"), source.fps,
                                 (frame.shape[1], frame.shape[0] + PANEL_HEIGHT))
    frames, state, frame_no, last_part_time, start = [], "empty", 0, 0.0, time.perf_counter()
    source_name = "camera" if source.is_camera else Path(args.source).name
    print(f"Running on {args.source}: station {roi}, {args.votes} frame votes per part, "
          f"{MODEL_NAMES[args.model]} on {line.device}. Writing {line.out}/ ...")
    while ok:
        if args.reverse:
            frame = cv2.flip(frame, 1)
        line_time = time.perf_counter() - start if source.is_camera else frame_no / source.fps
        part, score_now, decide_now = trigger.update(find_parts(frame, roi, args.dark))
        if score_now:
            image = crop_part(frame, part)
            frames.append((image, line.score_frame(image)))
            state = "collecting"
        if decide_now and frames:
            last = line.finish_part(frames, source_name)
            state, frames, last_part_time = last["verdict"], [], line_time
        if part is None and not trigger.decided:  # the decided part has left too: station empty
            state = "empty"
        if line_time - last_part_time > line.monitor.stall_seconds:
            line.check_stall(line_time - last_part_time)
        annotated = line.annotate(frame, roi, state, part)
        if writer:
            writer.write(annotated)
        if args.display:
            cv2.imshow("LineGuard", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.max_parts and line.n_parts >= args.max_parts:
            break
        ok, frame = source.read()
        frame_no += 1
    if writer:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    set_seed()
    try:
        source = open_source(args.source)
        line = Line(args)
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}")
    try:
        if isinstance(source, FolderSource):
            run_folder(line, source)
        else:
            run_video(line, source)
    finally:
        source.release()
    counts = line.log.counts()
    print(f"\nDone: {line.n_parts} parts this run; QC log now holds {counts['total']} "
          f"({counts['PASS']} PASS, {counts['REVIEW']} REVIEW, {counts['REJECT']} REJECT). "
          f"Reject signals fired: {line.reject.count}. Outputs in {line.out}/")
    line.log.close()


if __name__ == "__main__":
    main()
