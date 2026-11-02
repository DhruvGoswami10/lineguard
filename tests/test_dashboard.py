"""Dashboard tests (T21): the Streamlit page renders from a QC log without errors, shows
the live verdicts and KPIs, and its settings form writes the settings table that the
line runner reads. Uses Streamlit's headless test harness, no browser."""
import sys
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.dashboard import windowed_rates
from src.qclog import QCLog

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "src" / "dashboard.py"


def seed_log(path: Path) -> None:
    log = QCLog(path)
    log.set_settings(t_low=11.0, t_high=12.0, moderate_pct=5, severe_pct=15, baseline_rate=0.05, window=20,
                     stall_seconds=10)
    verdicts = ["PASS"] * 6 + ["REJECT", "REVIEW", "PASS", "REJECT"]
    for i, verdict in enumerate(verdicts, start=1):
        score = {"PASS": 10.5, "REVIEW": 11.5, "REJECT": 12.8}[verdict]
        log.record(part_id=f"T-{i:04d}", model="foundad", model_version="foundad:deadbeef", score=score,
                   t_low=11.0, t_high=12.0, verdict=verdict, severity="moderate" if verdict == "REJECT" else None,
                   frames=5, source="line_demo.mp4", image_path="none.png", heatmap_path="none.png")
    log.add_alert("rate_spike", "Reject rate 40% over the last 5 parts exceeds the control limit 34%", 0.4)
    log.close()


class TestDashboard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "qc.sqlite"
        seed_log(self.db)
        self.argv = sys.argv
        sys.argv = ["dashboard.py", "--db", str(self.db)]

    def tearDown(self):
        sys.argv = self.argv
        self.tmp.cleanup()

    def render(self) -> AppTest:
        at = AppTest.from_file(str(PAGE), default_timeout=120)
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    def test_kpis_and_latest_verdict(self):
        at = self.render()
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual((metrics["Inspected"], metrics["PASS"], metrics["REVIEW"], metrics["REJECT"]),
                         ("10", "7", "1", "2"))
        self.assertEqual(metrics["Reject rate"], "20.0%")
        self.assertTrue(any("REJECT (moderate)" in m.value for m in at.markdown))   # the latest part
        self.assertTrue(any("rate_spike" in e.value for e in at.error))             # the alert
        self.assertEqual(len(at.button), 3)                                         # Save + Good/Defect for the REVIEW part

    def test_saving_settings_writes_the_table(self):
        at = self.render()
        inputs = {n.label: n for n in at.number_input}
        inputs["REJECT from (t_high)"].set_value(13.5)
        inputs["stall alert after (s)"].set_value(30.0)
        next(b for b in at.button if b.label == "Save settings").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("Saved" in s.value for s in at.success))
        log = QCLog(self.db)
        settings = log.get_settings()
        log.close()
        self.assertEqual((settings["t_high"], settings["stall_seconds"]), (13.5, 30.0))

    def test_review_confirmation_stores_the_operator_label(self):
        at = self.render()
        good = next(b for b in at.button if b.label == "Confirm good")
        good.click().run()
        log = QCLog(self.db)
        labelled = [r for r in log.all_inspections() if r["operator_label"]]
        log.close()
        self.assertEqual(len(labelled), 1)
        self.assertEqual((labelled[0]["verdict"], labelled[0]["operator_label"]), ("REVIEW", "good"))

    def test_missing_log_shows_a_hint_not_an_error(self):
        sys.argv = ["dashboard.py", "--db", str(Path(self.tmp.name) / "missing.sqlite")]
        at = AppTest.from_file(str(PAGE), default_timeout=60)
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("run_line" in w.value for w in at.warning))

    def test_windowed_rates(self):
        verdicts = ["REJECT", "PASS", "PASS", "PASS", "REJECT", "REJECT", "PASS"]
        self.assertEqual(windowed_rates(verdicts, 3), [1 / 3, 2 / 3])   # the trailing part is not a full window


if __name__ == "__main__":
    unittest.main()
