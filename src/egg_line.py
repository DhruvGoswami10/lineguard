"""The egg line: finding, cropping and tracking eggs in a real conveyor recording.

A second product for the line, added 2026-11-04 per Dhruv for the demo video. The camera
is a fixed overhead IP camera over a rod conveyor on a poultry farm (a public YouTube clip,
see EGG_CLIP). Eggs enter at the top of the frame and roll downwards. The same FoundAD
detector inspects them; only the part finder is new, because the bottle line's finder (a
dark part on a bright belt, src/acquisition.py) does not fit brown eggs on a striped belt.

What is product- and camera-specific, and why:

- **The inspection band.** The camera vendor's counting software draws a line across the
  frame at rows 350-351 and boxes / numbers on every egg below it. Only the band above
  that line is clean footage, so only eggs whose whole crop sits in it are inspected.
- **The egg finder.** Eggs are warm-coloured (red well above blue); the belt and floor are
  grey or bluish. "Warmth" = red - blue of the blurred frame. Strongly warm pixels seed the
  eggs; the egg then grows into weakly warm pixels, but only up to one egg radius from its
  centre. Why two levels: a white or pale egg (the clearest anomaly in the clip) is only
  strongly warm in places, and a single low cut would also pick up straw on the belt.
- **Touching eggs** (they pile up against the left rail) are split by giving every warm
  pixel to the nearest egg centre, centres being the peaks of the distance transform.
- **Egg size depends on position**: the lens is a fisheye, so eggs near the image centre
  look bigger. expected_size() is a fit measured once on 134 isolated eggs.

Each egg is cropped as a square of CROP_SCALE x its expected size, centred on it, so every
crop shows the egg at the same scale in the middle; FoundAD then scores only the centred
disc (foreground rule "disc", src/foundad.py), not the moving rods around it.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from src.decision import vote
from src.foundad import DISC_RADIUS

EGG_CLIP = {
    "file": "data/egg_line/1EgUxp6BlBY.mp4",
    "url": "https://www.youtube.com/watch?v=1EgUxp6BlBY",
    "title": "Egg Counting with IP Camera on Conveyor Belt",
    "channel": "CamContador",
    "sha256": "a5be1d1b0a1029c5c4cc0271d89cd6995b708b11960f0b23303cb9c66a7b81cd",  # 1080p, 20 fps, 179.8 s
}

SEED_WARMTH = 18     # red - blue above this: certainly egg (brown eggs sit at 40-90, the belt at -15..5)
GROW_WARMTH = 4      # red - blue above this, near a seed: still egg (a white egg's paler half sits at 2-16)
BLUR_SIGMA = 3       # smooths the video's 8x8 compression blocks before measuring colour
CROP_SCALE = 1.3     # crop side / expected egg size: the egg's short half-axis (~0.33 of the side)
                     # stays outside the scored disc (DISC_RADIUS = 0.27) even if the centre is ~13 px off
MIN_SEED_DEPTH = 25  # px: a seed must be at least this thick (distance-transform peak) to start an egg
# A new egg is found as it enters: its top edge is then at most ~0.15 egg sizes into the frame
# (one was first found at 0.15, frame 3200; a rule of "still cut by the frame top" missed it).
# An egg that already has 3 votes is at least ~0.45 egg sizes in, so a track born deeper than
# ENTRY_DEPTH is a re-found egg and is not inspected again (ready()).
ENTRY_DEPTH = 0.3

# expected_size(): egg axes (px) as a quadratic in the distance from the optical centre,
# least-squares fit on 134 isolated eggs (every 10th frame of the clip, 4 Nov 2026).
SIZE_FIT_MINOR = (200.2, -0.144, 3.22e-5)
SIZE_FIT_MAJOR = (279.7, -0.295, 1.42e-4)


@dataclass(frozen=True)
class Layout:
    """Where things are in this camera's 1920 x 1080 frame."""
    band_bottom: int = 348               # last clean row: the vendor's count line is at rows 350-351
    # The vendor's box round each counted egg starts above the egg, so it reaches up into the
    # band: box tops measured at rows 274-285 (frames 629, 1377). The scored disc of an egg
    # must end above them; the crop's margin may still show a box line (it is not scored).
    disc_bottom: int = 268
    x_min: int = 200                     # left rail and the eggs piled on it
    x_max: int = 1760                    # right rail
    # the vendor's on-screen text (x0, y0, x1, y1): egg count (top left; its red digits reach
    # x 239, y 113 once the count has 4 digits, measured over the clip), timestamp (top right)
    osd: tuple = ((0, 0, 250, 120), (1430, 40, 1870, 110))
    centre: tuple = (960, 540)           # optical centre, for the fisheye size model


LAYOUT = Layout()


@dataclass
class Egg:
    """One egg in one frame: fitted ellipse (centre, axes in px, angle in degrees)."""
    x: float
    y: float
    major: float
    minor: float
    angle: float
    expected: float                      # expected size here (expected_size), sets the crop

    def crop_box(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) of the square crop centred on the egg."""
        half = CROP_SCALE * self.expected / 2
        return (int(round(self.x - half)), int(round(self.y - half)),
                int(round(self.x + half)), int(round(self.y + half)))


def expected_size(x: float, y: float, layout: Layout = LAYOUT) -> float:
    """Mean of the expected long and short axes of an egg centred at (x, y)."""
    r = float(np.hypot(x - layout.centre[0], y - layout.centre[1]))
    minor = SIZE_FIT_MINOR[0] + SIZE_FIT_MINOR[1] * r + SIZE_FIT_MINOR[2] * r * r
    major = SIZE_FIT_MAJOR[0] + SIZE_FIT_MAJOR[1] * r + SIZE_FIT_MAJOR[2] * r * r
    return (minor + major) / 2


def warmth(frame: np.ndarray) -> np.ndarray:
    """Red minus blue of the blurred BGR frame (int16): high on eggs, near zero on the belt."""
    blurred = cv2.GaussianBlur(frame, (0, 0), BLUR_SIGMA).astype(np.int16)
    return blurred[..., 2] - blurred[..., 0]


def _open(mask: np.ndarray, size: int) -> np.ndarray:
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))


def _seed_centres(strong: np.ndarray, layout: Layout) -> list[tuple[float, float]]:
    """Egg centres from the strongly warm mask: peaks of its distance transform, strongest
    first, keeping a peak only if it is more than 0.6 egg sizes from every kept one."""
    depth = cv2.distanceTransform(strong, cv2.DIST_L2, 5)
    peaks = (depth == cv2.dilate(depth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))) \
        & (depth > MIN_SEED_DEPTH)
    ys, xs = np.nonzero(peaks)
    centres: list[tuple[float, float]] = []
    for _, x, y in sorted(zip(depth[ys, xs], xs, ys), reverse=True):
        if all(np.hypot(x - cx, y - cy) > 0.6 * expected_size(cx, cy, layout) for cx, cy in centres):
            centres.append((float(x), float(y)))
    return centres


def _regions(grow: np.ndarray, centres: list[tuple[float, float]], layout: Layout) -> list[np.ndarray | None]:
    """Each warm pixel goes to its nearest centre, if within 0.6 egg sizes of it; returns
    each egg's outer contour (None if it got no pixels)."""
    h, w = grow.shape
    best = np.full((h, w), np.inf, np.float32)
    owner = np.full((h, w), -1, np.int32)
    for i, (cx, cy) in enumerate(centres):
        r = 0.6 * expected_size(cx, cy, layout)
        x0, x1 = max(int(cx - r), 0), min(int(cx + r) + 1, w)
        y0, y1 = max(int(cy - r), 0), min(int(cy + r) + 1, h)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
        win_best, win_owner = best[y0:y1, x0:x1], owner[y0:y1, x0:x1]
        closer = (d2 < win_best) & (d2 <= r * r)
        win_best[closer], win_owner[closer] = d2[closer], i
    contours = []
    for i in range(len(centres)):
        mask = ((owner == i) & (grow > 0)).astype(np.uint8)
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.append(max(found, key=cv2.contourArea) if found else None)
    return contours


def find_eggs(frame: np.ndarray, layout: Layout = LAYOUT) -> list[Egg]:
    """All eggs in the clean band of a BGR frame (including ones not fully visible)."""
    band = frame[:layout.band_bottom + 1]
    heat = warmth(band)
    strong = _open(heat > SEED_WARMTH, 9)
    grow = _open(heat > GROW_WARMTH, 7) | strong
    centres = _seed_centres(strong, layout)
    for _ in range(2):  # re-centre on each region's centroid: a pale egg's seed is off-centre
        contours = _regions(grow, centres, layout)
        moved = []
        for c in contours:
            if c is not None and len(c) >= 5:
                m = cv2.moments(c)
                if m["m00"] > 0:
                    moved.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
        centres = moved
    eggs = []
    for c in _regions(grow, centres, layout) if centres else []:
        if c is None or len(c) < 5:
            continue
        (x, y), axes, angle = cv2.fitEllipse(cv2.convexHull(c))
        eggs.append(Egg(x, y, max(axes), min(axes), angle, expected_size(x, y, layout)))
    return eggs


def inspectable(egg: Egg, layout: Layout = LAYOUT) -> bool:
    """True if the egg can be inspected in this frame: its whole crop is inside the clean
    band and between the rails, its scored disc is above the vendor's boxes, it is clear
    of the vendor's on-screen text, and its fitted size is plausible for an egg at that
    position (not a fragment or two merged eggs)."""
    x0, y0, x1, y1 = egg.crop_box()
    if x0 < layout.x_min or x1 > layout.x_max or y0 < 0 or y1 > layout.band_bottom:
        return False
    if egg.y + DISC_RADIUS * (x1 - x0) > layout.disc_bottom:
        return False
    for ox0, oy0, ox1, oy1 in layout.osd:
        if x0 < ox1 and x1 > ox0 and y0 < oy1 and y1 > oy0:
            return False
    return 0.7 < (egg.major + egg.minor) / 2 / egg.expected < 1.35


def ready(track: "Track", layout: Layout = LAYOUT) -> bool:
    """True if this frame of the track may be inspected: the egg is inspectable here, and the
    track began near the top of the frame (top edge less than ENTRY_DEPTH egg sizes in). A
    track that begins lower down is an egg the tracker lost and found again after it could
    already have been decided; inspecting it would decide that egg a second time. Added
    2026-11-04 per Dhruv after the first full run counted one egg twice."""
    return track.entered and inspectable(track.egg, layout)


def crop_egg(frame: np.ndarray, egg: Egg) -> Image.Image:
    """The square crop centred on the egg (RGB PIL image), edge-padded if it leaves the frame."""
    x0, y0, x1, y1 = egg.crop_box()
    h, w = frame.shape[:2]
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return Image.fromarray(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))


@dataclass(eq=False)  # tracks are compared by identity, not by field values
class Track:
    """One egg followed through the frames."""
    id: int
    egg: Egg
    vx: float = 0.0
    vy: float = 0.0
    missed: int = 0                                   # frames in a row without a match
    entered: bool = True                              # first seen near the frame top (ENTRY_DEPTH)
    seen_now: bool = True
    scores: list[float] = field(default_factory=list)  # one per inspectable frame, until decided
    score: float | None = None                        # median of the votes, once decided
    number: int | None = None                         # inspection number ("egg #n"), once decided
    crop: Image.Image | None = None                   # latest scored crop and its heatmap (for display)
    heatmap: np.ndarray | None = None


class EggTracker:
    """Follows eggs from frame to frame and collects one score per clean frame.

    Matching: each track predicts its next position from its velocity (eggs roll down at
    ~11-15 px per frame at 20 fps) and takes the nearest unmatched egg within MATCH_FRACTION
    of an egg size that is not higher up the frame than BACKSTEP_FRACTION of an egg size
    (eggs only roll down: without this, a track whose egg went undetected for a frame took
    over the egg above it, and its own egg came back as a new track; 4 Nov run, eggs #203 and
    #205). Unmatched eggs start new tracks; a track unseen for MAX_MISSED frames is dropped.
    A track is decided as soon as it has `votes` scores (median, the same vote as the bottle
    line), so the verdict appears while the egg is still on screen.
    """
    MATCH_FRACTION = 0.45
    BACKSTEP_FRACTION = 0.1   # detection noise and jostling in a pile move a centre up a few px at most
    MAX_MISSED = 2
    START_VELOCITY = (0.0, 11.0)

    def __init__(self, votes: int = 5):
        self.votes = votes
        self.tracks: list[Track] = []
        self.next_id = 1
        self.decided = 0

    def update(self, eggs: list[Egg]) -> list[Track]:
        """Match this frame's eggs to tracks; returns the tracks seen in this frame."""
        pairs = []
        for t in self.tracks:
            px, py = t.egg.x + t.vx, t.egg.y + t.vy
            for j, e in enumerate(eggs):
                if e.y < t.egg.y - self.BACKSTEP_FRACTION * e.expected:
                    continue                               # would mean rolling back up the belt
                d = float(np.hypot(e.x - px, e.y - py))
                if d <= self.MATCH_FRACTION * e.expected:
                    pairs.append((d, t, j))
        used_tracks, used_eggs = set(), set()
        for t in self.tracks:
            t.seen_now = False
        for _, t, j in sorted(pairs, key=lambda p: p[0]):
            if t.id in used_tracks or j in used_eggs:
                continue
            e = eggs[j]
            t.vx, t.vy = 0.5 * t.vx + 0.5 * (e.x - t.egg.x), 0.5 * t.vy + 0.5 * (e.y - t.egg.y)
            t.egg, t.missed, t.seen_now = e, 0, True
            used_tracks.add(t.id)
            used_eggs.add(j)
        for j, e in enumerate(eggs):
            if j not in used_eggs:
                entered = e.y - e.expected / 2 < ENTRY_DEPTH * e.expected
                self.tracks.append(Track(self.next_id, e, *self.START_VELOCITY, entered=entered))
                self.next_id += 1
        for t in self.tracks:
            if not t.seen_now:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.MAX_MISSED]
        return [t for t in self.tracks if t.seen_now]

    def add_score(self, track: Track, score: float) -> bool:
        """Record one frame's score; returns True when this vote decides the egg."""
        if track.score is not None:
            return False
        track.scores.append(score)
        return len(track.scores) >= self.votes and self._decide(track)

    def close(self, track: Track, min_votes: int) -> bool:
        """The egg can get no more clean frames (it left the band, or was lost): decide it
        on the frames it has if there are at least min_votes; returns True if decided."""
        return track.score is None and len(track.scores) >= min_votes and self._decide(track)

    def _decide(self, track: Track) -> bool:
        track.score = vote(track.scores)
        self.decided += 1
        track.number = self.decided
        return True
