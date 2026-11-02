"""QC log tests (T20): every verdict is stored with score, time and image path; settings,
alerts, operator labels and CSV export round-trip. Uses a temporary SQLite file."""
import csv
import tempfile
import unittest
from pathlib import Path

from src.qclog import INSPECTION_COLUMNS, QCLog


def record(log: QCLog, part: str, score: float, verdict: str, severity=None) -> int:
    return log.record(part_id=part, model="foundad", model_version="foundad:abc12345", score=score,
                      t_low=11.0, t_high=12.0, verdict=verdict, severity=severity, frames=5,
                      source="line_demo.mp4", image_path=f"parts/{part}.png", heatmap_path=f"parts/{part}_heatmap.png")


class TestQCLog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = QCLog(Path(self.tmp.name) / "qc.sqlite")

    def tearDown(self):
        self.log.close()
        self.tmp.cleanup()

    def test_every_field_is_stored_and_returned(self):
        row_id = record(self.log, "P1", 12.5, "REJECT", "moderate")
        row = self.log.recent(1)[0]
        self.assertEqual(row["id"], row_id)
        for key in INSPECTION_COLUMNS:
            self.assertIn(key, row)
        self.assertEqual((row["part_id"], row["score"], row["verdict"], row["severity"], row["frames"]),
                         ("P1", 12.5, "REJECT", "moderate", 5))
        self.assertTrue(row["timestamp"].startswith("20"))  # ISO-8601 UTC
        self.assertEqual(row["image_path"], "parts/P1.png")

    def test_counts_and_reject_rate(self):
        for part, score, verdict in (("a", 1, "PASS"), ("b", 1, "PASS"), ("c", 1, "REVIEW"), ("d", 2, "REJECT")):
            record(self.log, part, score, verdict)
        counts = self.log.counts()
        self.assertEqual((counts["PASS"], counts["REVIEW"], counts["REJECT"], counts["total"]), (2, 1, 1, 4))
        self.assertAlmostEqual(counts["reject_rate"], 0.25)

    def test_empty_log_counts(self):
        counts = self.log.counts()
        self.assertEqual(counts["total"], 0)
        self.assertEqual(counts["reject_rate"], 0.0)
        self.assertIsNone(self.log.last_timestamp())

    def test_recent_is_newest_first_and_filterable(self):
        record(self.log, "first", 1, "PASS")
        record(self.log, "second", 2, "REJECT")
        record(self.log, "third", 1, "REVIEW")
        self.assertEqual([r["part_id"] for r in self.log.recent(2)], ["third", "second"])
        self.assertEqual([r["part_id"] for r in self.log.recent(10, verdict="REVIEW")], ["third"])

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ValueError):
            self.log.record(part_id="x", model="ae", model_version="v", score=1.0, t_low=1, t_high=1,
                            verdict="PASS", verdcit="typo")

    def test_bad_verdict_is_rejected_by_the_schema(self):
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            self.log.record(part_id="x", model="ae", model_version="v", score=1.0, t_low=1, t_high=1,
                            verdict="MAYBE")

    def test_settings_round_trip_and_overwrite(self):
        self.assertEqual(self.log.get_settings(), {})
        self.log.set_settings(t_low=1.5, t_high=2.0)
        self.log.set_settings(t_high=2.5, window=20)
        self.assertEqual(self.log.get_settings(), {"t_low": 1.5, "t_high": 2.5, "window": 20.0})

    def test_operator_label(self):
        row_id = record(self.log, "P1", 11.5, "REVIEW")
        self.log.set_operator_label(row_id, "good", "scratch on the belt, not the bottle")
        row = self.log.recent(1)[0]
        self.assertEqual((row["operator_label"], row["operator_note"]), ("good", "scratch on the belt, not the bottle"))

    def test_alerts(self):
        self.log.add_alert("rate_spike", "Reject rate 40% exceeds limit 20%", 0.4)
        alerts = self.log.alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual((alerts[0]["kind"], alerts[0]["value"]), ("rate_spike", 0.4))

    def test_csv_export_matches_the_table(self):
        record(self.log, "P1", 10.0, "PASS")
        record(self.log, "P2", 13.0, "REJECT", "severe")
        path = Path(self.tmp.name) / "out" / "qc.csv"
        self.assertEqual(self.log.export_csv(path), 2)
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["part_id"] for r in rows], ["P1", "P2"])
        self.assertEqual(rows[1]["severity"], "severe")
        self.assertEqual(float(rows[0]["score"]), 10.0)

    def test_two_connections_see_the_same_data(self):
        """The runner writes while the dashboard reads the same file."""
        record(self.log, "P1", 10.0, "PASS")
        other = QCLog(self.log.path)
        self.assertEqual(other.counts()["total"], 1)
        other.set_settings(t_high=99.0)
        self.assertEqual(self.log.get_settings()["t_high"], 99.0)
        other.close()


if __name__ == "__main__":
    unittest.main()
