"""Acquisition tests (T19): the ROI, finding and cropping the part in synthetic frames,
the presence trigger's state machine, and the folder source. No model needed."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.acquisition import (ROI, FolderSource, PartLocation, PresenceTrigger, crop_part, find_parts,
                             open_source)

W, H = 640, 360
STATION = ROI(120, 30, 400, 300)


def frame_with_discs(centres: list[tuple[int, int]], radius: int = 60) -> np.ndarray:
    """A white frame (backlit belt) with dark discs (parts)."""
    frame = np.full((H, W, 3), 255, np.uint8)
    for cx, cy in centres:
        cv2.circle(frame, (cx, cy), radius, (40, 30, 30), -1)
    return frame


class TestROI(unittest.TestCase):
    def test_parse_and_clip(self):
        self.assertEqual(ROI.parse("10,20,300,200"), ROI(10, 20, 300, 200))
        self.assertEqual(ROI(-5, 10, 1000, 1000).clip(640, 360), ROI(0, 10, 640, 350))
        with self.assertRaises(ValueError):
            ROI.parse("10,20")


class TestFindParts(unittest.TestCase):
    def test_empty_station(self):
        self.assertEqual(find_parts(frame_with_discs([]), STATION), [])

    def test_part_fully_inside(self):
        parts = find_parts(frame_with_discs([(320, 180)]), STATION)
        self.assertEqual(len(parts), 1)
        p = parts[0]
        self.assertTrue(p.fully_inside)
        self.assertAlmostEqual(p.centre[0], 320, delta=2)
        self.assertAlmostEqual(p.centre[1], 180, delta=2)
        self.assertAlmostEqual(p.w, 120, delta=3)

    def test_entering_and_leaving_flags(self):
        entering = find_parts(frame_with_discs([(130, 180)]), STATION)[0]   # overlaps the left edge
        leaving = find_parts(frame_with_discs([(515, 180)]), STATION)[0]    # overlaps the right edge
        self.assertTrue(entering.touches_left and not entering.touches_right)
        self.assertTrue(leaving.touches_right and not leaving.touches_left)
        self.assertFalse(entering.fully_inside)

    def test_small_specks_are_ignored(self):
        parts = find_parts(frame_with_discs([(320, 180)], radius=5), STATION)
        self.assertEqual(parts, [])

    def test_two_parts_largest_first(self):
        frame = frame_with_discs([(200, 180)], radius=50)
        cv2.circle(frame, (430, 180), 70, (0, 0, 0), -1)
        parts = find_parts(frame, STATION)
        self.assertEqual(len(parts), 2)
        self.assertGreater(parts[0].area, parts[1].area)


class TestCrop(unittest.TestCase):
    def test_crop_is_square_centred_with_dataset_margin(self):
        frame = frame_with_discs([(320, 180)], radius=60)
        part = find_parts(frame, STATION)[0]
        image = crop_part(frame, part)
        self.assertEqual(image.size[0], image.size[1])
        self.assertAlmostEqual(image.size[0], round(part.w * 900 / 800), delta=1)
        arr = np.asarray(image)
        centre = arr[arr.shape[0] // 2, arr.shape[1] // 2]
        self.assertLess(centre.mean(), 60)       # the part is in the middle ...
        self.assertEqual(int(arr[0, 0].mean()), 255)  # ... with white around it

    def test_crop_off_frame_is_padded_white(self):
        frame = frame_with_discs([(40, 40)], radius=120)
        part = find_parts(frame, ROI(0, 0, W, H))[0]
        image = np.asarray(crop_part(frame, part))
        self.assertEqual(image.shape[0], image.shape[1])
        self.assertEqual(int(image[0, 0].mean()), 255)


def moving_blobs(frame_no: int, n_parts: int = 3, speed: int = 10, spacing: int = 800,
                 width: int = 560, roi=ROI(240, 60, 800, 600)) -> list[PartLocation]:
    """The geometry of the demo video: parts of `width` px, `spacing` apart, moving right."""
    out = []
    for i in range(n_parts):
        x = -width - i * spacing + frame_no * speed
        x0, x1 = max(x, roi.x), min(x + width, roi.x + roi.w)
        if x1 > x0:
            out.append(PartLocation(x0, roi.y, x1 - x0, roi.h, (x1 - x0) * roi.h,
                                    touches_left=x0 <= roi.x, touches_right=x1 >= roi.x + roi.w))
    return out


class TestPresenceTrigger(unittest.TestCase):
    def test_one_decision_per_part_with_n_votes(self):
        trigger, decisions, scored = PresenceTrigger(votes=5), [], 0
        for f in range(400):
            part, score, decide = trigger.update(moving_blobs(f))
            scored += score
            if decide:
                decisions.append(f)
        self.assertEqual(len(decisions), 3)       # three parts, three decisions ...
        self.assertEqual(scored, 15)              # ... five scored frames each
        self.assertEqual(decisions[0], 85)        # fully inside from frame 81: votes on frames 81-85

    def test_parts_closer_than_the_station_are_still_separated(self):
        """With 800 px spacing and an 800 px station the next part enters as the last leaves."""
        trigger, decisions = PresenceTrigger(votes=3), 0
        for f in range(400):
            decisions += trigger.update(moving_blobs(f, spacing=620))[2]
        self.assertEqual(decisions, 3)

    def test_part_leaving_early_is_decided_on_the_frames_seen(self):
        trigger = PresenceTrigger(votes=10)
        inside = PartLocation(500, 60, 300, 600, 1, False, False)
        leaving = PartLocation(900, 60, 300, 600, 1, False, True)
        for _ in range(4):
            self.assertEqual(trigger.update([inside])[1:], (True, False))
        self.assertEqual(trigger.update([leaving])[1:], (False, True))   # decide on the 4 frames
        self.assertEqual(trigger.update([leaving])[1:], (False, False))  # and not again
        self.assertEqual(trigger.update([])[1:], (False, False))

    def test_entering_part_is_not_scored(self):
        trigger = PresenceTrigger(votes=2)
        entering = PartLocation(240, 60, 100, 600, 1, True, False)
        self.assertEqual(trigger.update([entering])[1:], (False, False))


class TestSources(unittest.TestCase):
    def test_folder_source_replays_each_image_once_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("b.png", "a.png", "notes.txt"):
                if name.endswith(".png"):
                    cv2.imwrite(str(Path(tmp) / name), frame_with_discs([]))
                else:
                    (Path(tmp) / name).write_text("x")
            source = open_source(tmp)
            self.assertIsInstance(source, FolderSource)
            self.assertEqual(len(source), 2)
            names = []
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                names.append(source.current_name)
                self.assertEqual(frame.shape, (H, W, 3))
            self.assertEqual(names, ["a.png", "b.png"])

    def test_missing_video_gives_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            open_source("no_such_video.mp4")


if __name__ == "__main__":
    unittest.main()
