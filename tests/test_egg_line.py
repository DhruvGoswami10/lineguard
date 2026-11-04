"""Egg line (src/egg_line.py, src/egg_demo.py): egg finder, inspection band, crop, tracker
and display, on synthetic frames drawn here, so no clip and no network are needed.
Added 2026-11-04 per Dhruv. The end-to-end run at the bottom needs weights/foundad_egg.pt
and the DINOv2 file, like the bottle line's end-to-end test."""
import argparse
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

from src.decision import REJECT, Rule
from src.download_backbone import backbone_path
from src.egg_demo import FPS, PANEL_HEIGHT, EggRun, read_exclusions
from src.egg_line import (CROP_SCALE, LAYOUT, Egg, EggTracker, Track, crop_egg, expected_size, find_eggs,
                          inspectable, ready)
from src.qclog import QCLog

REPO = Path(__file__).resolve().parents[1]
BROWN, BELT, ROD = (70, 100, 160), (140, 140, 140), (90, 80, 76)   # BGR
HAVE_EGG_ASSETS = (REPO / "weights" / "foundad_egg.pt").exists() and (REPO / backbone_path("vits14")).exists()


def belt(height: int = 1080, width: int = 1920) -> np.ndarray:
    """Grey belt with dark horizontal rods every 45 px, like the rod conveyor."""
    frame = np.full((height, width, 3), BELT, np.uint8)
    for y in range(20, height, 45):
        frame[y:y + 10] = ROD
    return frame


def draw_egg(frame: np.ndarray, x: int, y: int, colour=BROWN, pale_colour=None, cap: float = 0.45) -> tuple[int, int]:
    """A filled egg of the expected size at (x, y). With pale_colour, only the top `cap` of
    its height has `colour`, the rest is pale_colour: a pale egg that is strongly warm only in
    a cap, as the white egg in the clip. Returns the drawn half-axes."""
    e = expected_size(x, y)
    axes = (int(e * 0.56), int(e * 0.44))                 # long : short ~ 1.25, mean = e
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.ellipse(mask, (x, y), axes, 0, 0, 360, 1, -1)
    frame[mask > 0] = colour
    if pale_colour is not None:
        lower = mask.copy()
        lower[: y - axes[1] + int(cap * 2 * axes[1])] = 0
        frame[lower > 0] = pale_colour
    return axes


def near(eggs, x, y, tol=12):
    return [e for e in eggs if abs(e.x - x) <= tol and abs(e.y - y) <= tol]


class TestEggFinder(unittest.TestCase):
    def test_finds_brown_eggs_at_their_centres_and_size(self):
        frame = belt()
        for x in (500, 960, 1400):
            draw_egg(frame, x, 160)
        eggs = find_eggs(frame)
        self.assertEqual(len(eggs), 3)
        for x in (500, 960, 1400):
            (egg,) = near(eggs, x, 160)
            self.assertAlmostEqual((egg.major + egg.minor) / 2 / egg.expected, 1.0, delta=0.12)

    def test_pale_egg_comes_out_whole(self):
        """Hysteresis: seeded in the strongly warm cap (red-blue 30, top 45%), grown into the
        weakly warm rest (red-blue 10), so the egg is one blob at its true centre. With one
        cut at the seed level only the cap would be found, centred too high and too small."""
        frame = belt()
        axes = draw_egg(frame, 900, 170, colour=(170, 185, 200), pale_colour=(185, 190, 195))
        (egg,) = near(find_eggs(frame), 900, 170, tol=15)
        self.assertGreater(egg.minor, 0.85 * 2 * axes[1])  # covers (almost) the full height

    def test_touching_eggs_are_split(self):
        frame = belt()
        axes = draw_egg(frame, 700, 170)
        draw_egg(frame, 700 + int(1.8 * axes[0]), 170)    # long axes overlap by 10%
        self.assertEqual(len(find_eggs(frame)), 2)

    def test_straw_on_the_belt_is_not_an_egg(self):
        frame = belt()
        cv2.line(frame, (800, 150), (930, 190), (60, 170, 200), 6)   # warm, but thin
        self.assertEqual(find_eggs(frame), [])

    def test_nothing_below_the_clean_band_is_looked_at(self):
        frame = belt()
        draw_egg(frame, 960, 520)                         # in the vendor's counting zone
        self.assertEqual(find_eggs(frame), [])

    def test_eggs_look_bigger_near_the_optical_centre(self):
        self.assertGreater(expected_size(960, 200), expected_size(300, 200))


class TestInspectionBand(unittest.TestCase):
    def egg(self, x, y, scale=1.0):
        e = expected_size(x, y)
        return Egg(x, y, 1.12 * e * scale, 0.88 * e * scale, 0.0, e)

    def test_egg_fully_in_the_clean_band_is_inspected(self):
        self.assertTrue(inspectable(self.egg(960, 150)))

    def test_crop_crossing_the_vendors_count_line_is_not(self):
        egg = self.egg(960, 280)
        self.assertGreater(egg.crop_box()[3], LAYOUT.band_bottom)
        self.assertFalse(inspectable(egg))

    def test_disc_reaching_the_vendors_boxes_is_not(self):
        egg = self.egg(960, 215)                          # crop still ends above row 348 ...
        self.assertLessEqual(egg.crop_box()[3], LAYOUT.band_bottom)
        self.assertFalse(inspectable(egg))                # ... but its scored disc reaches row 268

    def test_eggs_cut_by_the_frame_top_at_the_rails_or_under_text_are_not(self):
        self.assertFalse(inspectable(self.egg(960, 40)))      # crop above row 0
        self.assertFalse(inspectable(self.egg(150, 160)))     # left rail
        self.assertFalse(inspectable(self.egg(1700, 160)))    # right rail / under the timestamp
        self.assertFalse(inspectable(self.egg(260, 140)))     # under the vendor's egg counter

    def test_four_digit_vendor_counter_is_avoided(self):
        """Bug found 4 Nov: once the vendor's count had 4 digits its red digits (to x 239)
        showed in the crops of eggs near the left rail."""
        self.assertFalse(inspectable(self.egg(320, 160)))    # crop overlaps the counter's area
        self.assertTrue(inspectable(self.egg(320, 205)))     # same column, below the counter

    def test_fragments_and_merged_eggs_are_not(self):
        self.assertFalse(inspectable(self.egg(960, 150, scale=0.5)))
        self.assertFalse(inspectable(self.egg(960, 150, scale=1.6)))


class TestCrop(unittest.TestCase):
    def test_crop_is_square_centred_and_scaled_to_the_egg(self):
        frame = belt()
        draw_egg(frame, 960, 160)
        egg = near(find_eggs(frame), 960, 160)[0]
        crop = np_rgb(crop_egg(frame, egg))
        self.assertEqual(crop.shape[0], crop.shape[1])
        self.assertAlmostEqual(crop.shape[0], CROP_SCALE * egg.expected, delta=2)
        c = crop.shape[0] // 2
        self.assertEqual(tuple(crop[c, c]), BROWN[::-1])  # the egg is in the middle (RGB)

    def test_crop_leaving_the_frame_is_padded_not_shrunk(self):
        frame = belt()
        egg = Egg(30, 30, 100, 80, 0, 100)
        self.assertEqual(crop_egg(frame, egg).size, (130, 130))


def np_rgb(image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


class TestTracker(unittest.TestCase):
    def frames(self, n, start_y=100, step=11.0):
        """Two eggs rolling down side by side at the clip's speed."""
        for k in range(n):
            y = start_y + step * k
            yield [Egg(500, y, 160, 130, 0, 150), Egg(1200, y + 5, 160, 130, 0, 150)]

    def test_ids_stay_with_their_eggs(self):
        tracker = EggTracker()
        for eggs in self.frames(8):
            seen = tracker.update(eggs)
        self.assertEqual(sorted(t.id for t in seen), [1, 2])
        self.assertEqual({round(t.egg.x) for t in seen}, {500, 1200})

    def test_decides_on_the_median_of_five_votes(self):
        tracker, track = EggTracker(votes=5), Track(1, Egg(0, 0, 1, 1, 0, 1))
        decided = [tracker.add_score(track, s) for s in (1.0, 9.0, 2.0, 3.0, 2.5)]
        self.assertEqual(decided, [False, False, False, False, True])
        self.assertEqual(track.score, 2.5)                 # one odd frame (9.0) does not decide it
        self.assertEqual(track.number, 1)
        self.assertFalse(tracker.add_score(track, 100.0))  # decided once only

    def test_early_close_needs_enough_votes(self):
        tracker = EggTracker(votes=5)
        short, enough = Track(1, Egg(0, 0, 1, 1, 0, 1)), Track(2, Egg(0, 0, 1, 1, 0, 1))
        short.scores, enough.scores = [1.0, 2.0], [1.0, 2.0, 4.0]
        self.assertFalse(tracker.close(short, min_votes=3))
        self.assertTrue(tracker.close(enough, min_votes=3))
        self.assertEqual(enough.score, 2.0)

    def test_lost_eggs_are_dropped_and_new_ones_get_new_ids(self):
        tracker = EggTracker()
        for eggs in self.frames(3):
            tracker.update(eggs)
        for _ in range(EggTracker.MAX_MISSED + 1):
            tracker.update([])
        self.assertEqual(tracker.tracks, [])
        (track,) = tracker.update([Egg(800, 100, 160, 130, 0, 150)])
        self.assertEqual(track.id, 3)

    def test_a_track_does_not_take_over_the_egg_above_it(self):
        """Bug found 4 Nov (eggs #203 and #205 of the first run, frames 3204-3211): the egg was
        not detected for two frames, its track took over the egg entering above it (37 px
        higher), and the egg came back as a new track and was decided twice. Replayed here
        with the clip's positions."""
        tracker = EggTracker()
        (a,) = tracker.update([Egg(358, 137, 160, 130, 0, 140)])
        tracker.update([Egg(355, 139, 160, 130, 0, 140)])
        tracker.update([Egg(321, 102, 160, 130, 0, 140)])  # egg A undetected; egg B appears above it
        tracker.update([Egg(313, 111, 160, 130, 0, 140)])
        seen = tracker.update([Egg(316, 121, 160, 130, 0, 140), Egg(358, 192, 160, 130, 0, 140)])
        (back,) = [t for t in seen if t.egg.y == 192]
        self.assertEqual(back.id, a.id)                    # egg A is still track A
        (b,) = [t for t in seen if t.egg.y == 121]
        self.assertNotEqual(b.id, a.id)

    def test_a_track_that_starts_mid_band_is_never_inspected(self):
        """Backstop for the same bug: an egg first seen below the frame top is one the tracker
        lost and found again, so it is not inspected (and not decided) a second time."""
        tracker = EggTracker()
        (entering,) = tracker.update([Egg(960, 40, 180, 150, 0, 167)])   # crop still crosses row 0
        self.assertTrue(entering.entered)
        (found_again,) = EggTracker().update([Egg(960, 150, 180, 150, 0, 167)])
        self.assertFalse(found_again.entered)
        self.assertTrue(inspectable(found_again.egg))     # it would be inspectable ...
        self.assertFalse(ready(found_again))              # ... but it is not inspected

    def test_an_egg_found_just_inside_the_frame_is_still_inspected(self):
        """Bug found 4 Nov (second run): an egg with a dark mark was first detected with its
        crop 0.5 px inside the frame (frame 3200, x 381, y 82), and a rule of 'still cut by the
        frame top' never inspected it. Its top edge was 0.15 egg sizes in: a new egg."""
        (track,) = EggTracker().update([Egg(381, 82, 139, 111, 0, expected_size(381, 82))])
        self.assertGreaterEqual(track.egg.crop_box()[1], 0)   # the old rule would have refused it
        self.assertTrue(track.entered)


class TestDisplayAndFiles(unittest.TestCase):
    def test_exclusion_list_is_read_by_file_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "excluded.csv"
            path.write_text("file,reason\nf00001_e0001.png,white egg\n")
            self.assertEqual(read_exclusions(str(path)), {"f00001_e0001.png": "white egg"})
            self.assertEqual(read_exclusions(str(Path(tmp) / "none.csv")), {})

    def test_committed_exclusions_name_a_reason_each(self):
        with open(REPO / "splits" / "egg_line_excluded.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(rows)
        self.assertTrue(all(r["file"].endswith(".png") and r["reason"].strip() for r in rows))

    def test_frame_renders_with_outline_chip_and_panel(self):
        run = object.__new__(EggRun)                       # display state only, no model
        run.rule, run.map_floor = Rule(t_low=1.0, t_high=2.0), 0.5
        run.counts = {"PASS": 3, "REVIEW": 0, "REJECT": 1}
        run.ckpt = {"config": {"n_train": 100}}
        run.fonts = {s: ImageFont.load_default(size=s) for s in (14, 16, 18, 22, 26, 30, 44)}
        track = Track(7, Egg(960, 160, 180, 150, 0, 167))
        info = {"number": 4, "verdict": REJECT, "severity": "moderate", "score": 3.5, "votes": 5, "time_s": 1.0,
                "overlay": np.full((448, 448, 3), 200, np.uint8)}
        run.results, run.latest_flagged = {7: info}, info
        canvas = run.render(belt(), [track])
        self.assertEqual(canvas.shape, (LAYOUT.band_bottom + 1 + PANEL_HEIGHT, 1920, 3))
        chip = canvas[160 - 75 - 34 + 14, 960]            # middle of the chip above the egg (BGR)
        self.assertEqual(tuple(chip), (68, 68, 239))       # REJECT red


@unittest.skipUnless(HAVE_EGG_ASSETS, "weights/foundad_egg.pt or the DINOv2 file missing")
class TestEggRunEndToEnd(unittest.TestCase):
    """A 3-second synthetic clip of eggs rolling down the belt through the whole run command:
    every egg that gets votes is decided once, logged, and the video has every frame."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cls.clip, cls.n_frames = tmp / "eggs.mp4", 3 * FPS
        writer = cv2.VideoWriter(str(cls.clip), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 1080))
        for k in range(cls.n_frames):
            frame = belt()
            for x, y0 in ((600, 40), (1000, -60), (1300, 20)):
                draw_egg(frame, x, int(y0 + 11 * k))
            writer.write(frame)
        writer.release()
        cls.out = tmp / "run"
        done = subprocess.run([sys.executable, "-m", "src.egg_demo", "run", "--clip", str(cls.clip), "--start", "0",
                               "--out", str(cls.out)], cwd=REPO, capture_output=True, text=True, timeout=900)
        if done.returncode != 0:
            raise AssertionError(f"egg_demo run failed:\n{done.stdout}\n{done.stderr}")
        with open(cls.out / "inspections.csv", newline="") as f:
            cls.rows = list(csv.DictReader(f))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_each_egg_is_decided_once(self):
        self.assertEqual(len(self.rows), 3)
        self.assertEqual(len({r["track"] for r in self.rows}), 3)
        self.assertEqual([int(r["number"]) for r in self.rows], [1, 2, 3])

    def test_every_decision_is_in_the_qc_log(self):
        log = QCLog(self.out / "qc_log.sqlite")
        try:
            self.assertEqual(len(log.all_inspections()), len(self.rows))
        finally:
            log.close()

    def test_review_sheets_show_every_decided_egg(self):
        done = subprocess.run([sys.executable, "-m", "src.egg_demo", "review", "--out", str(self.out)], cwd=REPO,
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn(f"{len(self.rows)} crops", done.stdout)
        self.assertTrue((self.out / "media" / "review" / "sheet_01.jpg").is_file())

    def test_annotated_video_has_every_frame(self):
        cap = cv2.VideoCapture(str(self.out / "media" / "egg_line_demo.mp4"))
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.n_frames)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), LAYOUT.band_bottom + 1 + PANEL_HEIGHT)
        cap.release()


if __name__ == "__main__":
    unittest.main()
