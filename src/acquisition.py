"""Acquisition (subsystems 1 and 2): frames in, one centred image per part out.

Three sources, one interface (`open_source`):
    a video file      samples/line_demo.mp4   the recorded line used for the demo
    a camera          0, 1, ...               a webcam / USB camera, read live
    a folder          samples/                replay: each image file is one part

Finding the part (non-AI, by design): the belt is backlit, so the part is the dark
blob inside the inspection station (the ROI). Its bounding box gives a square crop
centred on the part, with the same margin around it as the dataset photographs,
so what the model sees on the line matches what it was trained on. A part counts as
present only when it is fully inside the station (its blob touches neither the left
nor the right edge), which is what a photocell trigger on a real line checks.

PresenceTrigger is the state machine: EMPTY -> PRESENT (collect up to N frames to
vote on) -> DONE (the decided part is leaving) -> EMPTY. One part, one decision.
Parts travel left to right (mirror the frame with --reverse for the other way), so a
blob at the left edge while the decided part is still leaving is the next part.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
DATASET_MARGIN = 900 / 800   # MVTec bottle photos: ~800 px part in a 900 px frame
DEFAULT_DARK = 160           # grey level below which a pixel belongs to the part (Otsu on the dataset gave ~160)
MIN_AREA_FRACTION = 0.05     # a blob smaller than this share of the ROI is noise, not a part


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def parse(cls, text: str) -> "ROI":
        """'x,y,w,h' from the command line."""
        try:
            x, y, w, h = (int(v) for v in text.split(","))
        except ValueError as err:
            raise ValueError(f"--roi must be x,y,w,h (got {text!r})") from err
        return cls(x, y, w, h)

    def clip(self, width: int, height: int) -> "ROI":
        x, y = max(0, self.x), max(0, self.y)
        return ROI(x, y, min(self.w, width - x), min(self.h, height - y))


@dataclass(frozen=True)
class PartLocation:
    """Where the part is in the frame (pixel coordinates of the full frame)."""
    x: int
    y: int
    w: int
    h: int
    area: int
    touches_left: bool    # still entering the station
    touches_right: bool   # already leaving it

    @property
    def fully_inside(self) -> bool:
        return not (self.touches_left or self.touches_right)

    @property
    def centre(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------
class FolderSource:
    """Replay a folder of images as if each one were a captured part (no trigger needed)."""

    def __init__(self, folder: Path):
        self.paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        self.index = 0
        self.fps = 1.0

    def read(self):
        if self.index >= len(self.paths):
            return False, None
        frame = cv2.imread(str(self.paths[self.index]))
        self.index += 1
        return frame is not None, frame

    @property
    def current_name(self) -> str:
        return self.paths[self.index - 1].name

    def release(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.paths)


class VideoSource:
    """A video file or a camera index, through OpenCV."""

    def __init__(self, spec: str):
        self.is_camera = spec.isdigit()
        self.cap = cv2.VideoCapture(int(spec) if self.is_camera else spec)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video source {spec!r}"
                                    + ("" if self.is_camera else " (file missing or not a video?)"))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_camera else 0

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()


def open_source(spec: str):
    path = Path(spec)
    if path.is_dir():
        return FolderSource(path)
    return VideoSource(spec)


# ----------------------------------------------------------------------------
# Finding and cropping the part
# ----------------------------------------------------------------------------
def find_parts(frame: np.ndarray, roi: ROI, dark: int = DEFAULT_DARK,
               min_area_fraction: float = MIN_AREA_FRACTION) -> list[PartLocation]:
    """Dark blobs inside the ROI that are big enough to be a part, largest first."""
    grey = cv2.cvtColor(frame[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w], cv2.COLOR_BGR2GRAY)
    mask = (grey < dark).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    parts = []
    for label in range(1, n):  # label 0 is the background
        x, y, w, h, area = (int(v) for v in stats[label])
        if area >= min_area_fraction * roi.w * roi.h:
            parts.append(PartLocation(roi.x + x, roi.y + y, w, h, area,
                                      touches_left=x <= 0, touches_right=x + w >= roi.w))
    return sorted(parts, key=lambda p: -p.area)


def crop_part(frame: np.ndarray, part: PartLocation, margin: float = DATASET_MARGIN) -> Image.Image:
    """Square crop centred on the part, padded with white where it runs off the frame."""
    side = int(round(max(part.w, part.h) * margin))
    cx, cy = part.centre
    x0, y0 = cx - side // 2, cy - side // 2
    canvas = np.full((side, side, 3), 255, np.uint8)
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + side, frame.shape[1]), min(y0 + side, frame.shape[0])
    if fx1 > fx0 and fy1 > fy0:
        canvas[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] = frame[fy0:fy1, fx0:fx1]
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


# ----------------------------------------------------------------------------
# Trigger
# ----------------------------------------------------------------------------
class PresenceTrigger:
    """Decides which frames to score. One part, one decision.

    Parts travel left to right and never overtake, so the next part in line is always
    the right-most blob that has not been decided yet. After a decision the trigger
    remembers the decided part's x and follows it rightwards until it leaves the
    station (several decided parts can be in view at once); every blob behind them
    is undecided. This copes with parts closer together than the station is wide,
    which is where a plain "wait until the station is empty" rule fails.
    """

    EMPTY, PRESENT = "EMPTY", "PRESENT"
    BACKSTEP = 20  # px a blob's left edge may jitter backwards and still be the same part

    def __init__(self, votes: int = 5):
        self.votes = max(1, votes)
        self.state = self.EMPTY
        self.collected = 0
        self.decided = []       # left edges of decided parts still in view, right-most first

    def _skip_decided(self, parts: list[PartLocation]) -> list[PartLocation]:
        """Match the remembered decided parts to this frame's blobs (both right-most first);
        a remembered part with no matching blob has left. Returns the undecided blobs."""
        still_here, i = [], 0
        for x in self.decided:
            if i < len(parts) and parts[i].x >= x - self.BACKSTEP:
                still_here.append(parts[i].x)
                i += 1
        self.decided = still_here
        return parts[i:]

    def update(self, parts: list[PartLocation]) -> tuple[PartLocation | None, bool, bool]:
        """Feed this frame's detections. Returns (the part being handled, score it?, decide now?)."""
        parts = self._skip_decided(sorted(parts, key=lambda p: -p.x))  # right-most first
        if not parts:
            self.state, self.collected = self.EMPTY, 0
            return None, False, False
        part = parts[0]
        if not part.fully_inside:
            if self.state == self.PRESENT and self.collected:
                self._done(part)                   # it left early: decide on the frames we have
                return part, False, True
            return part, False, False              # entering: wait until it is fully visible
        self.state = self.PRESENT
        self.collected += 1
        if self.collected >= self.votes:
            self._done(part)
            return part, True, True                # the last vote: score it, then decide
        return part, True, False

    def _done(self, part: PartLocation) -> None:
        self.state, self.collected = self.EMPTY, 0
        self.decided.append(part.x)  # it is behind every part decided before it
