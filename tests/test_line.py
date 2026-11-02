"""End-to-end tests (T23): a short conveyor video is rendered from real sample images,
the line runner processes it on CPU, and the QC log, reject events, saved crops and
annotated video are checked against the video's manifest.

Needs weights/foundad.pt, the DINOv2 file and samples/ (no dataset). Takes ~1 minute on CPU."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2

from src.download_backbone import backbone_path
from src.qclog import QCLog

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "samples"
HAVE_ASSETS = ((REPO / "weights" / "foundad.pt").exists() and (REPO / backbone_path("vits14")).exists()
               and SAMPLES.is_dir())
VIDEO_IMAGES = ["good_003.png", "broken_large_002.png", "good_018.png", "contamination_005.png"]


def run(args: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    done = subprocess.run([sys.executable, "-m", *args], cwd=REPO, capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        raise AssertionError(f"{' '.join(args)} failed:\n{done.stdout}\n{done.stderr}")
    return done


@unittest.skipUnless(HAVE_ASSETS, "weights/foundad.pt, the DINOv2 file or samples/ missing")
class TestLineEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        images = tmp / "images"
        images.mkdir()
        for name in VIDEO_IMAGES:
            (images / name).write_bytes((SAMPLES / name).read_bytes())
        cls.video = tmp / "line_test.mp4"
        run(["src.make_line_video", "--images", str(images), "--out", str(cls.video), "--speed", "20"])
        cls.manifest = json.loads(cls.video.with_suffix(".manifest.json").read_text())
        cls.out = tmp / "line"
        cls.run_output = run(["src.run_line", "--source", str(cls.video), "--model", "foundad", "--votes", "3",
                              "--out", str(cls.out), "--reset", "--quiet"]).stdout
        cls.log = QCLog(cls.out / "qc_log.sqlite")

    @classmethod
    def tearDownClass(cls):
        cls.log.close()
        cls.tmp.cleanup()

    def test_manifest_lists_every_image_once(self):
        self.assertEqual(sorted(p["image"] for p in self.manifest["parts"]), sorted(VIDEO_IMAGES))
        for part in self.manifest["parts"]:
            self.assertIsNotNone(part["enters_station"])
            self.assertGreater(part["leaves_station"], part["enters_station"])

    def test_one_decision_per_part_in_order(self):
        rows = self.log.all_inspections()
        self.assertEqual(len(rows), len(self.manifest["parts"]))
        self.assertEqual([r["frames"] for r in rows], [3] * len(rows))        # the requested votes
        self.assertEqual([r["part_id"] for r in rows], sorted(r["part_id"] for r in rows))

    def test_every_decision_is_logged_with_its_evidence(self):
        for row in self.log.all_inspections():
            self.assertIn(row["verdict"], ("PASS", "REVIEW", "REJECT"))
            self.assertTrue(Path(row["image_path"]).is_file(), row["image_path"])
            self.assertTrue(Path(row["heatmap_path"]).is_file(), row["heatmap_path"])
            self.assertEqual(row["source"], self.video.name)
            self.assertTrue(row["model_version"].startswith("foundad:"))
            self.assertLessEqual(row["t_low"], row["t_high"])
            expected = "REJECT" if row["score"] >= row["t_high"] else ("REVIEW" if row["score"] >= row["t_low"] else "PASS")
            self.assertEqual(row["verdict"], expected)

    def test_reject_signal_fired_once_per_reject(self):
        events = (self.out / "reject_events.jsonl").read_text().splitlines() \
            if (self.out / "reject_events.jsonl").exists() else []
        self.assertEqual(len(events), self.log.counts()["REJECT"])
        for line in events:
            self.assertEqual(json.loads(line)["action"], "DIVERT")

    def test_verdicts_agree_with_the_labels_in_the_manifest(self):
        """The model's job, on parts it never trained on: good passes, defects do not pass."""
        rows = self.log.all_inspections()
        for part, row in zip(self.manifest["parts"], rows):
            if part["label"] == "good":
                self.assertEqual(row["verdict"], "PASS", part["image"])
            else:
                self.assertNotEqual(row["verdict"], "PASS", part["image"])

    def test_annotated_video_has_one_frame_per_input_frame(self):
        cap = cv2.VideoCapture(str(self.out / "line_test_annotated.mp4"))
        self.assertTrue(cap.isOpened())
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.manifest["frames"])
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), self.manifest["height"] + 200)
        cap.release()

    def test_settings_were_seeded_from_the_checkpoint(self):
        settings = self.log.get_settings()
        for key in ("t_low", "t_high", "moderate_pct", "severe_pct", "baseline_rate", "window", "stall_seconds"):
            self.assertIn(key, settings)

    def test_new_product_can_be_commissioned_from_good_photos(self):
        """train_foundad --good-dir: a few good photos, no labels, thresholds from their spread."""
        tmp = Path(self.tmp.name)
        photos = tmp / "my_product"
        photos.mkdir()
        for name in ("good_000.png", "good_002.png", "good_003.png", "good_007.png"):
            (photos / name).write_bytes((SAMPLES / name).read_bytes())
        out = tmp / "weights_my_product" / "foundad.pt"
        done = run(["src.train_foundad", "--good-dir", str(photos), "--shots", "2", "--size", "224", "--steps", "2",
                    "--log-every", "1", "--device", "cpu", "--out", str(out), "--results", str(tmp / "r")])
        self.assertIn("Label-free calibration on 2 held-out good photos", done.stdout)
        import torch
        ckpt = torch.load(out, map_location="cpu", weights_only=True)
        self.assertEqual(ckpt["kind"], "foundad")
        self.assertLessEqual(ckpt["config"]["t_low"], ckpt["threshold"])
        self.assertEqual(ckpt["config"]["n_train"], 2)
        self.assertTrue(ckpt["config"]["train_source"].endswith("my_product"))

    def test_folder_replay_mode(self):
        out = Path(self.tmp.name) / "replay"
        run(["src.run_line", "--source", "samples", "--model", "foundad", "--out", str(out), "--reset",
             "--quiet", "--max-parts", "2"])
        log = QCLog(out / "qc_log.sqlite")
        rows = log.all_inspections()
        log.close()
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["source"] for r in rows], ["broken_large_002.png", "broken_large_003.png"])


if __name__ == "__main__":
    unittest.main()
