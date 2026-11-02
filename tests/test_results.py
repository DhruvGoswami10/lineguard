"""Checks that the committed results pack is complete and internally consistent:
SUMMARY.md must quote the same numbers as metrics.csv, and the confusion matrix
in metrics.csv must match the per-image verdicts in scores_split_b.csv."""
import csv
import unittest
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
EXPECTED_FILES = ["SUMMARY.md", "metrics.csv", "scores_split_b.csv", "roc.png", "confusion_matrices.png"]
PER_MODEL_FILES = ["score_hist_{kind}.png", "{kind}_training_curve.png"]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@unittest.skipUnless((RESULTS / "metrics.csv").exists(), "run `python -m src.evaluate` first")
class TestResultsPack(unittest.TestCase):
    def setUp(self):
        self.metrics = {row["model"]: row for row in read_csv(RESULTS / "metrics.csv")}
        self.images = read_csv(RESULTS / "scores_split_b.csv")
        self.summary = (RESULTS / "SUMMARY.md").read_text(encoding="utf-8")

    def test_all_files_exist(self):
        for name in EXPECTED_FILES:
            self.assertTrue((RESULTS / name).is_file(), name)
        for kind in self.metrics:
            for pattern in PER_MODEL_FILES:
                self.assertTrue((RESULTS / pattern.format(kind=kind)).is_file(), pattern.format(kind=kind))

    def test_every_technique_is_in_the_pack(self):
        self.assertEqual(set(self.metrics), {"ae", "cnn", "foundad"})

    def test_six_heatmaps_per_model(self):
        for kind in self.metrics:
            self.assertEqual(len(list((RESULTS / "heatmaps").glob(f"{kind}_*.png"))), 6, kind)

    def test_summary_quotes_metrics_csv(self):
        for row in self.metrics.values():
            for key in ("auroc", "precision", "recall", "f1", "specificity"):
                self.assertIn(f"{float(row[key]):.3f}", self.summary, f"{row['model']} {key}")
            self.assertIn(f"{row['tp']} / {row['fn']} / {row['fp']} / {row['tn']}", self.summary)

    def test_confusion_matrix_matches_per_image_verdicts(self):
        self.assertEqual(len(self.images), 42)  # split B
        for kind, row in self.metrics.items():
            def count(label, verdict):
                return sum(i["label"] == label and i[f"{kind}_verdict"] == verdict for i in self.images)
            recount = (count("1", "REJECT"), count("1", "OK"), count("0", "REJECT"), count("0", "OK"))
            self.assertEqual(recount, (int(row["tp"]), int(row["fn"]), int(row["fp"]), int(row["tn"])), kind)

    def test_verdicts_follow_threshold(self):
        for kind, row in self.metrics.items():
            threshold = float(row["threshold"])
            for image in self.images:
                expected = "REJECT" if float(image[f"{kind}_score"]) >= threshold else "OK"
                self.assertEqual(image[f"{kind}_verdict"], expected)


if __name__ == "__main__":
    unittest.main()
