"""The egg-line demo: real conveyor footage, inspected egg by egg. Added 2026-11-04 per Dhruv.

    # 0. the clip (not in the repo: it is someone else's video). Any downloader works, e.g.
    #    yt-dlp -f 137 -o data/egg_line/1EgUxp6BlBY.mp4 https://www.youtube.com/watch?v=1EgUxp6BlBY
    python -m src.egg_demo crops --start 0 --end 105            # 1. one crop per egg (the first 105 s)
    python -m src.egg_demo sheets                               # 2. contact sheets to review them by eye
    python -m src.train_foundad --good-dir data/egg_line/good --shots 48 --steps 2000 \\
        --foreground disc --out weights/foundad_egg.pt --results results/egg_line   # 3. train + calibrate
    python -m src.egg_demo check                                # 3b. scores of held-out and excluded crops
    python -m src.egg_demo run --start 105                      # 4. inspect the rest, render the video
    python -m src.egg_demo review                               # 5. every decided egg on sheets, by score

Step 1 writes the crops of eggs listed in splits/egg_line_excluded.csv (eggs that look
abnormal when the contact sheets are reviewed) to data/egg_line/excluded/ instead, so the
model learns "normal" from normal eggs only. The two parts of the clip never overlap:
training and calibration use 0-105 s, the demo inspects 105 s to the end.

Step 4 writes results/egg_line/: inspections.csv (one row per inspected egg), summary.json,
a QC log the dashboard can open (streamlit run src/dashboard.py -- --db
results/egg_line/qc_log.sqlite), and under media/ (git-ignored: it is derived from the
clip) the annotated video and a heatmap of every egg sent to REVIEW or REJECT.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.common import set_seed
from src.decision import PASS, REJECT, REVIEW, decide, rule_from_checkpoint
from src.egg_line import EGG_CLIP, LAYOUT, EggTracker, crop_egg, find_eggs, ready
from src.inference import inspect, load_model
from src.qclog import QCLog, now_iso
from src.visualize import overlay

FPS = 20                      # the clip's frame rate
PANEL_HEIGHT = 371            # band (349 rows) + panel = 720 rows: a 1920 x 720 video
MIN_VOTES = 3                 # an egg that leaves the clean band early is decided on at least this many frames
CROP_FRAME = 3                # training crops: the egg's 3rd clean frame (the middle of a 5-frame vote)

# Display colours (RGB): one per verdict, one per severity band.
COLOURS = {"inspecting": (226, 232, 240), PASS: (45, 212, 191), REVIEW: (245, 158, 11), REJECT: (239, 68, 68)}
SEVERITY_COLOURS = {"minor": (250, 204, 21), "moderate": (249, 115, 22), "severe": (220, 38, 38)}
PANEL_BG, TEXT, MUTED = (17, 24, 39), (241, 245, 249), (148, 163, 184)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Egg-line demo on real conveyor footage.")
    p.add_argument("command", choices=("crops", "sheets", "check", "run", "review"))
    p.add_argument("--clip", default=EGG_CLIP["file"])
    p.add_argument("--start", type=float, default=None, help="seconds into the clip (crops: 0, run: 105)")
    p.add_argument("--end", type=float, default=None, help="seconds into the clip (crops: 105, run: the end)")
    p.add_argument("--crops", default="data/egg_line", help="crops: where good/ and excluded/ go")
    p.add_argument("--exclude", default="splits/egg_line_excluded.csv",
                   help="crops: eggs judged abnormal on the contact sheets (file,reason)")
    p.add_argument("--weights", default="weights/foundad_egg.pt")
    p.add_argument("--out", default="results/egg_line")
    p.add_argument("--votes", type=int, default=5, help="frames per egg; the median decides")
    p.add_argument("--max-seconds", type=float, default=None, help="run: stop after this much footage")
    return p.parse_args()


def open_clip(path: str, start_s: float) -> tuple[cv2.VideoCapture, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {path}. Download the clip first (see the top of src/egg_demo.py): "
                         f"{EGG_CLIP['url']}")
    first = int(round(start_s * FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    return cap, first


def read_exclusions(path: str) -> dict[str, str]:
    if not Path(path).is_file():
        return {}
    with open(path, newline="") as f:
        return {row["file"]: row["reason"] for row in csv.DictReader(f)}


def export_crops(args) -> None:
    """One crop per egg: its CROP_FRAME-th clean frame. Normal eggs go to good/, the ones
    listed in the exclusion file to excluded/. File names carry the frame and track id."""
    start, end = args.start or 0.0, args.end if args.end is not None else 105.0
    excluded = read_exclusions(args.exclude)
    good_dir, bad_dir = Path(args.crops) / "good", Path(args.crops) / "excluded"
    for d in (good_dir, bad_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()
    cap, frame_no = open_clip(args.clip, start)
    tracker, clean_frames, saved = EggTracker(), {}, [0, 0]
    while frame_no < end * FPS:
        ok, frame = cap.read()
        if not ok:
            break
        for track in tracker.update(find_eggs(frame)):
            if not ready(track):
                continue
            clean_frames[track.id] = clean_frames.get(track.id, 0) + 1
            if clean_frames[track.id] == CROP_FRAME:
                name = f"f{frame_no:05d}_e{track.id:04d}.png"
                bad = name in excluded
                crop_egg(frame, track.egg).save((bad_dir if bad else good_dir) / name)
                saved[bad] += 1
        frame_no += 1
    cap.release()
    print(f"{start:.0f}-{end:.0f} s: {saved[0]} good crops -> {good_dir}, {saved[1]} excluded -> {bad_dir}")


def check_calibration(args) -> None:
    """Score the training-segment crops the predictor did not train on: the held-out normal
    ones (what the thresholds were set from) and the ones excluded as abnormal by eye. Writes
    results/egg_line/calibration_check.json. A sanity check, not an evaluation: the excluded
    crops were picked by one reviewer from contact sheets, without the model's scores."""
    model, ckpt = load_model(args.weights, "cpu")
    trained = set(ckpt["config"]["train_images"])
    rule = rule_from_checkpoint(ckpt)
    groups = {"held_out_normal": sorted(p for p in (Path(args.crops) / "good").glob("*.png") if p.name not in trained),
              "excluded_by_eye": sorted((Path(args.crops) / "excluded").glob("*.png"))}
    reasons = read_exclusions(args.exclude)
    report = {"t_low": rule.t_low, "t_high": rule.t_high, "threshold_rule": ckpt["config"].get("threshold_rule")}
    for name, paths in groups.items():
        scores = np.array([inspect(model, ckpt, Image.open(p), "cpu")["score"] for p in paths])
        report[name] = {"n": len(scores), "mean": float(scores.mean()), "std": float(scores.std()),
                        "p99_5": float(np.percentile(scores, 99.5)), "max": float(scores.max()),
                        "at_or_above_t_low": int((scores >= rule.t_low).sum()),
                        "at_or_above_t_high": int((scores >= rule.t_high).sum())}
        if name == "excluded_by_eye":
            report[name]["crops"] = [{"file": p.name, "reason": reasons.get(p.name, ""), "score": round(float(s), 4),
                                      "verdict": decide(float(s), rule).verdict} for p, s in zip(paths, scores)]
        print(f"{name}: n={len(scores)}  mean {scores.mean():.3f}  sd {scores.std():.3f}  "
              f"99.5th pct {np.percentile(scores, 99.5):.3f}  >= review line {report[name]['at_or_above_t_low']}  "
              f">= reject line {report[name]['at_or_above_t_high']}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "calibration_check.json").write_text(json.dumps(report, indent=2))


def contact_sheets(args) -> None:
    """Grids of the training crops (file name under each) for reviewing them by eye."""
    files = sorted((Path(args.crops) / "good").glob("*.png"))
    write_sheets([(p, p.stem) for p in files], Path(args.out) / "media" / "sheets")


def review_sheets(args) -> None:
    """Every egg the run decided, highest score first, labelled '#n score verdict': for
    checking by eye which flagged eggs really look abnormal and which abnormal-looking
    eggs were passed."""
    out = Path(args.out)
    with open(out / "inspections.csv", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: -float(r["score"]))
    tiles = [(out / "media" / "eggs" / f"egg{int(r['number']):04d}.png",
              f"#{r['number']} {float(r['score']):.2f} {r['verdict']}") for r in rows]
    write_sheets(tiles, out / "media" / "review")


def write_sheets(tiles: list[tuple[Path, str]], out: Path, per_sheet: int = 100, tile: int = 128) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("sheet_*.jpg"):
        old.unlink()
    font = ImageFont.load_default(size=11)
    for s in range(0, len(tiles), per_sheet):
        chunk = tiles[s:s + per_sheet]
        sheet = Image.new("RGB", (10 * tile, ((len(chunk) + 9) // 10) * (tile + 14)), "black")
        draw = ImageDraw.Draw(sheet)
        for i, (path, label) in enumerate(chunk):
            x, y = (i % 10) * tile, (i // 10) * (tile + 14)
            sheet.paste(Image.open(path).convert("RGB").resize((tile, tile)), (x, y))
            draw.text((x + 2, y + tile), label, fill="white", font=font)
        sheet.save(out / f"sheet_{s // per_sheet + 1:02d}.jpg", quality=90)
    print(f"{len(tiles)} crops on {(len(tiles) + per_sheet - 1) // per_sheet} sheets in {out}")


class EggRun:
    """Inspects every egg in the clip segment and renders the annotated video."""

    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.media = self.out / "media"
        for sub in ("flagged", "eggs"):
            (self.media / sub).mkdir(parents=True, exist_ok=True)
            for old in (self.media / sub).glob("*.png"):
                old.unlink()
        self.model, self.ckpt = load_model(args.weights, "cpu")  # CPU: the calibrated scores
        self.rule = rule_from_checkpoint(self.ckpt)
        self.map_floor = self.ckpt["config"].get("map_floor", 0.0)
        db = self.out / "qc_log.sqlite"
        db.unlink(missing_ok=True)
        self.log = QCLog(db)
        self.tracker = EggTracker(args.votes)
        self.results: dict[int, dict] = {}     # track id -> decision
        self.rows: list[dict] = []
        self.latest_flagged: dict | None = None
        self.counts = {PASS: 0, REVIEW: 0, REJECT: 0}
        self.fonts = {s: ImageFont.load_default(size=s) for s in (14, 16, 18, 22, 26, 30, 44)}

    def score(self, frame: np.ndarray, track) -> None:
        crop = crop_egg(frame, track.egg)
        result = inspect(self.model, self.ckpt, crop, "cpu")
        track.crop, track.heatmap = crop, result["heatmap"]
        if self.tracker.add_score(track, result["score"]):
            self.finish(track)

    def finish(self, track) -> None:
        """Turn a decided track into a decision: verdict, severity, log row, display state."""
        d = decide(track.score, self.rule)
        sec = self.frame_no / FPS
        info = {"number": track.number, "track": track.id, "frame": self.frame_no, "time_s": round(sec, 2),
                "x": round(track.egg.x), "y": round(track.egg.y),
                "score": round(track.score, 5), "votes": len(track.scores), "verdict": d.verdict,
                "severity": d.severity or "", "frame_scores": " ".join(f"{s:.3f}" for s in track.scores)}
        track.crop.save(self.media / "eggs" / f"egg{track.number:04d}.png")  # every egg, for the review sheets
        image_path = ""
        if d.verdict != PASS:
            heat = (overlay(track.crop, track.heatmap, self.rule.t_high, self.map_floor) * 255).astype(np.uint8)
            info["overlay"] = heat
            image_path = str(self.media / "flagged" / f"egg{track.number:04d}_{d.verdict.lower()}.png")
            Image.fromarray(np.hstack([np.asarray(track.crop.resize((448, 448))), heat])).save(image_path)
            self.latest_flagged = info
        self.results[track.id] = info
        self.counts[d.verdict] += 1
        self.rows.append({k: v for k, v in info.items() if k != "overlay"})
        self.log.record(part_id=f"egg{track.number:04d}", timestamp=now_iso(), model="foundad",
                        model_version=Path(self.args.weights).name, score=track.score, t_low=self.rule.t_low,
                        t_high=self.rule.t_high, verdict=d.verdict, severity=d.severity, frames=len(track.scores),
                        source=f"{EGG_CLIP['url']} frame {self.frame_no}", image_path=image_path,
                        heatmap_path=image_path)

    def close_early(self, track) -> None:
        """An egg that leaves the clean band before `votes` frames: decide on what it has."""
        if self.tracker.close(track, MIN_VOTES):
            self.finish(track)

    def run(self) -> None:
        start = self.args.start if self.args.start is not None else 105.0
        cap, self.frame_no = open_clip(self.args.clip, start)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        end = self.args.end * FPS if self.args.end is not None else total
        if self.args.max_seconds:
            end = min(end, self.frame_no + self.args.max_seconds * FPS)
        video_path = self.media / "egg_line_demo.mp4"
        writer = None
        t0, first = time.perf_counter(), self.frame_no
        while self.frame_no < end:
            ok, frame = cap.read()
            if not ok:
                break
            before = list(self.tracker.tracks)
            seen = self.tracker.update(find_eggs(frame))
            for track in seen:
                if ready(track):
                    if track.score is None:
                        self.score(frame, track)
                elif track.scores:
                    self.close_early(track)
            alive = {t.id for t in self.tracker.tracks}
            for track in before:                       # tracks lost this frame
                if track.id not in alive:
                    self.close_early(track)
            canvas = self.render(frame, seen)
            if writer is None:
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                         (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
            self.frame_no += 1
            if (self.frame_no - first) % 200 == 0:
                print(f"  {(self.frame_no - first) / FPS:5.1f} s of footage, {sum(self.counts.values())} eggs "
                      f"decided ({time.perf_counter() - t0:.0f} s)", flush=True)
        cap.release()
        if writer is not None:
            writer.release()
        self.save(video_path, start, time.perf_counter() - t0)

    def save(self, video_path: Path, start: float, elapsed: float) -> None:
        with open(self.out / "inspections.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]) if self.rows else ["number"], lineterminator="\n")
            w.writeheader()
            w.writerows(self.rows)
        summary = {"clip": EGG_CLIP, "segment_s": [start, round(self.frame_no / FPS, 2)],
                   "weights": self.args.weights, "t_low": self.rule.t_low, "t_high": self.rule.t_high,
                   "votes": self.args.votes, "min_votes": MIN_VOTES, "inspected": sum(self.counts.values()),
                   **self.counts, "flagged": [r for r in self.rows if r["verdict"] != PASS],
                   "seconds": round(elapsed, 1)}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Inspected {summary['inspected']} eggs: PASS {self.counts[PASS]}, REVIEW {self.counts[REVIEW]}, "
              f"REJECT {self.counts[REJECT]}. Video: {video_path}  Log: {self.out / 'inspections.csv'}")

    # ------------------------------------------------------------------ drawing
    def render(self, frame: np.ndarray, seen) -> np.ndarray:
        """The clean band with an outline per egg, plus the panel underneath (BGR)."""
        band = frame[:LAYOUT.band_bottom + 1]
        canvas = np.zeros((band.shape[0] + PANEL_HEIGHT, band.shape[1], 3), np.uint8)
        canvas[:band.shape[0]] = band
        labels = []
        for track in seen:
            info = self.results.get(track.id)
            if info is None and not track.scores:
                continue                                 # never inspectable: not part of the inspection
            state = info["verdict"] if info else "inspecting"
            e = track.egg
            r, g, b = COLOURS[state]
            cv2.ellipse(canvas, ((e.x, e.y), (e.major, e.minor), e.angle), (b, g, r),
                        2 if state in ("inspecting", PASS) else 4, cv2.LINE_AA)
            if info and info["verdict"] != PASS:
                labels.append((e, info))
        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        for e, info in labels:
            self.chip(draw, e, info)
        self.header(draw)
        self.panel(image, draw, band.shape[0])
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def chip(self, draw, egg, info) -> None:
        text = info["verdict"] + (f" · {info['severity']}" if info["severity"] else "")
        font = self.fonts[18]
        w = draw.textlength(text, font=font) + 16
        x, y = egg.x - w / 2, max(egg.y - egg.minor / 2 - 34, 4)
        draw.rounded_rectangle((x, y, x + w, y + 28), radius=8, fill=COLOURS[info["verdict"]])
        draw.text((x + 8, y + 4), text, fill=(15, 23, 42), font=font)

    def header(self, draw) -> None:
        """Covers the camera vendor's egg counter with the system's own label."""
        x0, y0, x1, y1 = LAYOUT.osd[0]
        draw.rounded_rectangle((x0 + 6, y0 + 6, x1 - 4, y1 - 4), radius=10, fill=PANEL_BG)
        draw.text((x0 + 20, y0 + 22), "LineGuard", fill=TEXT, font=self.fonts[26])
        draw.text((x0 + 20, y0 + 60), "egg line", fill=COLOURS[PASS], font=self.fonts[16])

    def panel(self, image, draw, top: int) -> None:
        draw.rectangle((0, top, image.width, image.height), fill=PANEL_BG)
        f, info = self.fonts, self.latest_flagged
        # left: the latest egg sent to REVIEW or REJECT, with its heatmap
        if info is not None:
            inset = Image.fromarray(info["overlay"]).resize((290, 290))
            image.paste(inset, (24, top + 22))
            draw.rectangle((22, top + 20, 24 + 291, top + 22 + 291), outline=COLOURS[info["verdict"]], width=3)
        else:
            draw.rectangle((24, top + 22, 24 + 290, top + 22 + 290), outline=MUTED, width=2)
            draw.text((60, top + 155), "no anomaly yet", fill=MUTED, font=f[22])
        draw.text((24, top + 322), "latest anomaly (heatmap)", fill=MUTED, font=f[16])
        # middle: verdict, score against the two lines
        x = 350
        if info is not None:
            sev = info["severity"]
            title = info["verdict"] + (f" · {sev}" if sev else "")
            draw.text((x, top + 22), title, fill=COLOURS[info["verdict"]], font=f[44])
            draw.text((x, top + 80), f"egg #{info['number']}   ·   {info['time_s']:.1f} s into the clip   ·   "
                      f"median of {info['votes']} frames", fill=TEXT, font=f[22])
            self.score_bar(draw, x, top + 140, 860, info["score"])
        else:
            draw.text((x, top + 22), "watching for anomalies", fill=MUTED, font=f[44])
            self.score_bar(draw, x, top + 140, 860, None)
        # right: counters and the severity legend
        x = 1290
        n = sum(self.counts.values())
        draw.text((x, top + 22), f"{n} eggs inspected", fill=TEXT, font=f[30])
        for i, verdict in enumerate((PASS, REVIEW, REJECT)):
            draw.text((x + i * 200, top + 72), f"{verdict} {self.counts[verdict]}", fill=COLOURS[verdict], font=f[26])
        draw.text((x, top + 130), "severity (score above the REJECT line)", fill=MUTED, font=f[16])
        for i, (name, rng) in enumerate((("minor", "< +50%"), ("moderate", "+50-150%"), ("severe", "> +150%"))):
            y = top + 160 + i * 36
            draw.rounded_rectangle((x, y, x + 22, y + 22), radius=5, fill=SEVERITY_COLOURS[name])
            draw.text((x + 34, y), f"{name}   {rng}", fill=TEXT, font=f[18])
        cfg = self.ckpt["config"]
        draw.text((24, image.height - 26),
                  f"FoundAD: frozen DINOv2 ViT-S/14 + predictor trained on {cfg.get('n_train', '?')} normal eggs, "
                  f"no defect examples.   Footage: \"{EGG_CLIP['title']}\", {EGG_CLIP['channel']}, YouTube "
                  f"{EGG_CLIP['url'].split('=')[-1]} (only the band above the vendor's counting overlay).",
                  fill=MUTED, font=f[14])

    def score_bar(self, draw, x: int, y: int, width: int, score: float | None) -> None:
        """PASS | REVIEW | REJECT zones on one axis, with a marker at the score. The axis runs
        to 2.5 x the REJECT line, where the 'severe' band starts (+150%)."""
        t_low, t_high, font = self.rule.t_low, self.rule.t_high, self.fonts[16]
        top = max(2.5 * t_high, (score or 0) * 1.05)
        px = lambda v: x + width * min(v, top) / top
        zones = ((0, t_low, PASS), (t_low, t_high, REVIEW), (t_high, top, REJECT))
        for lo, hi, verdict in zones:
            if hi > lo:
                draw.rectangle((px(lo), y, px(hi), y + 22), fill=COLOURS[verdict])
        # zone names under the bar, skipped when a zone is too narrow to hold one
        for lo, hi, verdict in zones:
            if px(hi) - px(lo) >= draw.textlength(verdict, font=font) + 12:
                draw.text((px(lo) + 6, y + 30), verdict, fill=COLOURS[verdict], font=font)
        # the two lines' values above the bar: PASS line ends at its tick, REJECT line starts at its own
        low_text, high_text = f"review {t_low:.2f}", f"reject {t_high:.2f}"
        draw.text((px(t_low) - draw.textlength(low_text, font=self.fonts[14]) - 4, y - 22), low_text,
                  fill=MUTED, font=self.fonts[14])
        draw.text((px(t_high) + 4, y - 22), high_text, fill=MUTED, font=self.fonts[14])
        if score is not None:
            m = px(score)
            draw.polygon(((m - 10, y + 60), (m + 10, y + 60), (m, y + 26)), fill=TEXT)
            draw.text((m - 30, y + 64), f"score {score:.2f}", fill=TEXT, font=self.fonts[18])


def main() -> None:
    args = parse_args()
    set_seed()
    if args.command == "crops":
        export_crops(args)
    elif args.command == "sheets":
        contact_sheets(args)
    elif args.command == "review":
        review_sheets(args)
    elif args.command == "check":
        check_calibration(args)
    else:
        EggRun(args).run()


if __name__ == "__main__":
    main()
