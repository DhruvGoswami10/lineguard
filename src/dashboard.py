"""Operator dashboard (subsystem 7a): what the line is doing, read from the QC log.

    streamlit run src/dashboard.py                       # default QC log results/line/qc_log.sqlite
    streamlit run src/dashboard.py -- --db other.sqlite

Runs alongside src/run_line.py (which writes the log) and shows:
    - KPIs: parts inspected, PASS / REVIEW / REJECT counts, reject rate, line status
    - live view: the latest part, its heatmap, score and verdict
    - control chart: reject rate per window of N parts against the control limit
    - alerts raised by src/alerts.py
    - review queue: REVIEW parts waiting for a person, with Good / Defect buttons
      (the confirmed label is stored in the log, the seed of future retraining data)
    - settings: the thresholds, severity bands and alert limits. Saving them writes the
      settings table, which the line runner reads before every decision - no restart,
      no retraining.
Everything comes from the SQLite file, so the dashboard can run on another machine
that only sees the log.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `streamlit run src/dashboard.py` from the repo root
from src.alerts import control_limit  # noqa: E402
from src.decision import DEFAULT_SEVERITY_PCT  # noqa: E402
from src.qclog import DEFAULT_DB, QCLog  # noqa: E402

REFRESH_SECONDS = 2
COLOURS = {"PASS": "#4caf50", "REVIEW": "#ff9800", "REJECT": "#e53935"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DEFAULT_DB)
    return p.parse_known_args()[0]  # streamlit / unittest may pass arguments of their own


def seconds_since(iso: str | None) -> float | None:
    if not iso:
        return None
    return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()


def windowed_rates(verdicts: list[str], window: int) -> list[float]:
    """Reject rate of each consecutive block of `window` parts (the control-chart points)."""
    return [sum(v == "REJECT" for v in verdicts[i:i + window]) / window
            for i in range(0, len(verdicts) - window + 1, window)]


def control_chart(verdicts: list[str], settings: dict):
    window = int(settings.get("window", 20))
    baseline = settings.get("baseline_rate", 0.05)
    limit = control_limit(baseline, window, settings.get("sigma", 3.0))
    rates = windowed_rates(verdicts, window)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    if rates:
        ax.plot(range(1, len(rates) + 1), rates, marker="o", color="#455a64", label=f"reject rate per {window} parts")
    ax.axhline(limit, color="#e53935", linestyle="--", label=f"upper control limit {limit:.0%}")
    ax.axhline(baseline, color="#9e9e9e", linestyle=":", label=f"baseline {baseline:.0%}")
    ax.set(xlabel=f"window (x {window} parts)", ylabel="reject rate", ylim=(0, 1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def verdict_badge(verdict: str, severity: str | None) -> str:
    text = verdict + (f" ({severity})" if severity else "")
    return f"<span style='color:{COLOURS[verdict]};font-weight:700;font-size:1.6em'>{text}</span>"


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="LineGuard dashboard", layout="wide")
    st.title("LineGuard operator dashboard")
    if not Path(args.db).is_file():
        st.warning(f"No QC log at `{args.db}` yet. Start the line: "
                   "`python -m src.run_line --source samples/line_demo.mp4`")
        st.stop()
    live = st.sidebar.toggle("Live refresh", value=True, help=f"re-read the log every {REFRESH_SECONDS} s")
    st.sidebar.caption(f"QC log: `{args.db}`")

    @st.fragment(run_every=f"{REFRESH_SECONDS}s" if live else None)  # re-reads the log on a timer
    def body():
        page(args.db)

    body()


def page(db_path: str) -> None:
    """Everything below the title: re-run every few seconds while live refresh is on."""
    log = QCLog(db_path)
    settings = log.get_settings()
    counts = log.counts()
    recent = log.recent(200)

    # ---- KPIs ----------------------------------------------------------------
    since = seconds_since(log.last_timestamp())
    stall = settings.get("stall_seconds", 10.0)
    status = "no parts yet" if since is None else ("RUNNING" if since < stall else f"idle {since:.0f} s")
    k = st.columns(6)
    k[0].metric("Inspected", counts["total"])
    k[1].metric("PASS", counts["PASS"])
    k[2].metric("REVIEW", counts["REVIEW"])
    k[3].metric("REJECT", counts["REJECT"])
    k[4].metric("Reject rate", f"{counts['reject_rate']:.1%}")
    k[5].metric("Line status", status)

    # ---- live view + control chart -----------------------------------------
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Latest part")
        if recent:
            last = recent[0]
            st.markdown(verdict_badge(last["verdict"], last["severity"]), unsafe_allow_html=True)
            st.caption(f"{last['part_id']} · score {last['score']:.4g} · review ≥ {last['t_low']:.4g} · "
                       f"reject ≥ {last['t_high']:.4g} · {last['frames']} frame(s) · {last['model']} "
                       f"({last['model_version']}) · {last['timestamp'][11:19]} UTC")
            if last["heatmap_path"] and Path(last["heatmap_path"]).is_file():
                st.image(last["heatmap_path"], width="stretch")
            elif last["image_path"] and Path(last["image_path"]).is_file():
                st.image(last["image_path"], width=300)
        else:
            st.info("Waiting for the first part.")
    with right:
        st.subheader("Control chart")
        verdicts = [r["verdict"] for r in reversed(log.all_inspections())]
        st.pyplot(control_chart(verdicts, settings), clear_figure=True)
        alerts = log.alerts(5)
        st.subheader("Alerts")
        if alerts:
            for a in alerts:
                st.error(f"{a['timestamp'][11:19]} UTC · {a['kind']}: {a['message']}")
        else:
            st.success("No alerts.")

    # ---- review queue --------------------------------------------------------
    st.subheader("Review queue")
    queue = [r for r in log.recent(100, verdict="REVIEW") if not r["operator_label"]]
    if not queue:
        st.caption("Nothing waiting for review.")
    for r in queue[:5]:
        c1, c2 = st.columns([2, 1])
        with c1:
            if r["heatmap_path"] and Path(r["heatmap_path"]).is_file():
                st.image(r["heatmap_path"], width="stretch")
        with c2:
            st.write(f"**{r['part_id']}** · score {r['score']:.4g}")
            if st.button("Confirm good", key=f"good{r['id']}"):
                log.set_operator_label(r["id"], "good")
                st.rerun()
            if st.button("Confirm defect", key=f"defect{r['id']}"):
                log.set_operator_label(r["id"], "defect")
                st.rerun()

    # ---- settings ------------------------------------------------------------
    with st.expander("Line settings (applied to the next part, no retraining)"):
        with st.form("settings"):
            c = st.columns(4)
            t_low = c[0].number_input("REVIEW from (t_low)", value=float(settings.get("t_low", 0.0)), format="%.5f")
            t_high = c[1].number_input("REJECT from (t_high)", value=float(settings.get("t_high", 0.0)), format="%.5f")
            moderate = c[2].number_input("moderate from (% above t_high)", value=float(settings.get("moderate_pct", DEFAULT_SEVERITY_PCT[0])))
            severe = c[3].number_input("severe from (% above t_high)", value=float(settings.get("severe_pct", DEFAULT_SEVERITY_PCT[1])))
            c = st.columns(4)
            baseline = c[0].number_input("baseline reject rate", value=float(settings.get("baseline_rate", 0.05)),
                                         min_value=0.0, max_value=1.0, format="%.3f")
            window = c[1].number_input("control-chart window (parts)", value=int(settings.get("window", 20)), min_value=1)
            stall_s = c[2].number_input("stall alert after (s)", value=float(settings.get("stall_seconds", 10.0)))
            if st.form_submit_button("Save settings"):
                if t_low > t_high:
                    st.error("t_low must not exceed t_high.")
                else:
                    log.set_settings(t_low=t_low, t_high=t_high, moderate_pct=moderate, severe_pct=severe,
                                     baseline_rate=baseline, window=window, stall_seconds=stall_s)
                    st.success("Saved. The line runner uses these from its next part.")

    # ---- recent table + export -----------------------------------------------
    st.subheader("Recent inspections")
    if recent:
        rows = [{"id": r["id"], "part": r["part_id"], "time (UTC)": r["timestamp"][11:23], "score": round(r["score"], 5),
                 "verdict": r["verdict"], "severity": r["severity"] or "", "frames": r["frames"],
                 "operator": r["operator_label"] or ""} for r in recent[:50]]
        st.dataframe(rows, width="stretch", hide_index=True)
        export = Path(db_path).with_suffix(".csv")
        n = log.export_csv(export)
        st.download_button("Download the full log as CSV", data=export.read_bytes(), file_name=export.name,
                           mime="text/csv", help=f"{n} rows, also written to {export}")
    log.close()


if __name__ == "__main__":  # `streamlit run` executes the file as __main__
    main()
