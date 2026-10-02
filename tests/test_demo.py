"""Demo tests: the command runs on CPU, and its scores equal the reported evaluation scores.

Needs the committed weights/ and samples/ (no dataset)."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from src.data import load_image
from src.inference import inspect, load_model

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "samples"
HAVE_ASSETS = (REPO / "weights" / "ae.pt").exists() and SAMPLES.is_dir()


def original_path(sample: Path) -> str:
    """samples/broken_small_010.png -> test/broken_small/010.png"""
    folder, number = sample.stem.rsplit("_", 1)
    return f"test/{folder}/{number}.png"


@unittest.skipUnless(HAVE_ASSETS, "weights/ or samples/ missing")
class TestDemo(unittest.TestCase):
    def test_cli_runs_on_cpu_for_both_models(self):
        with tempfile.TemporaryDirectory() as out:
            for model in ("ae", "cnn"):
                done = subprocess.run(
                    [sys.executable, "-m", "src.demo", "--model", model, "--input", "samples/good_000.png",
                     "--device", "cpu", "--out", out],
                    cwd=REPO, capture_output=True, text=True, timeout=300)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertRegex(done.stdout, r"good_000\.png .* (OK|REJECT)")
                self.assertTrue((Path(out) / f"{model}_good_000.png").is_file())

    def test_folder_input_processes_every_sample(self):
        with tempfile.TemporaryDirectory() as out:
            done = subprocess.run([sys.executable, "-m", "src.demo", "--model", "ae", "--input", "samples",
                                   "--device", "cpu", "--out", out],
                                  cwd=REPO, capture_output=True, text=True, timeout=300)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(len(list(Path(out).glob("ae_*.png"))), len(list(SAMPLES.glob("*.png"))))

    def test_missing_input_gives_clean_error(self):
        done = subprocess.run([sys.executable, "-m", "src.demo", "--model", "ae", "--input", "no_such_folder"],
                              cwd=REPO, capture_output=True, text=True, timeout=300)
        self.assertNotEqual(done.returncode, 0)
        self.assertNotIn("Traceback", done.stderr)

    def test_samples_are_all_from_split_b(self):
        split_b = {i["path"] for i in json.loads((REPO / "splits" / "bottle_split.json").read_text())["split_b"]}
        for sample in SAMPLES.glob("*.png"):
            self.assertIn(original_path(sample), split_b, sample.name)

    @unittest.skipUnless((REPO / "results" / "scores_split_b.csv").exists(), "no evaluation results")
    def test_sample_scores_match_evaluation(self):
        with open(REPO / "results" / "scores_split_b.csv", encoding="utf-8", newline="") as f:
            reported = {row["path"]: row for row in csv.DictReader(f)}
        cpu = torch.device("cpu")
        for kind in ("ae", "cnn"):
            model, ckpt = load_model(REPO / "weights" / f"{kind}.pt", cpu)
            for sample in sorted(SAMPLES.glob("*.png")):
                result = inspect(model, ckpt, load_image(sample), cpu, with_heatmap=False)
                row = reported[original_path(sample)]
                self.assertAlmostEqual(result["score"], float(row[f"{kind}_score"]), places=5,
                                       msg=f"{kind} {sample.name}")
                self.assertEqual(result["verdict"], row[f"{kind}_verdict"])


if __name__ == "__main__":
    unittest.main()
