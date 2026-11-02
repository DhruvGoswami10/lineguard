"""Build the demo "camera feed": real bottle images moving along a backlit conveyor.

    python -m src.make_line_video                      # samples/ -> samples/line_demo.mp4
    python -m src.make_line_video --images my_photos/ --out my_line.mp4

Why a composited video: the proof of concept needs a camera feed that can be replayed
on any machine, that contains real parts with real defects, and that a marker can
re-create. The 20 held-out split-B photos from MVTec AD (good and defective bottles
nobody trained on) are placed one by one on a white, backlit belt and move under a
fixed camera, as on a bottle-inspection line. Nothing in the frame is generated: the
bottles are the dataset's photographs; only their motion is simulated.

The line runner (src/run_line.py) reads this file exactly as it would read a webcam,
so a real recording can replace it without code changes.

A manifest (JSON) next to the video records which image is which part and when it is
in view, for the end-to-end test only; the runner never reads it.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from src.common import SEED

WIDTH, HEIGHT = 1280, 720
LANE_TOP, LANE_BOTTOM = 60, 660        # the backlit belt between two dark rails
BOTTLE_PX = 560                        # size of each pasted 900x900 photo
STATION = (240, 60, 800, 600)          # x, y, w, h of the inspection station (the ROI)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a conveyor video from a folder of part photos.")
    p.add_argument("--images", default="samples", help="folder of part photos (square, white background)")
    p.add_argument("--out", default="samples/line_demo.mp4")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--speed", type=int, default=10, help="belt speed in pixels per frame")
    p.add_argument("--spacing", type=int, default=800, help="distance between parts in pixels")
    p.add_argument("--seed", type=int, default=SEED, help="order, jitter and rotation of the parts")
    p.add_argument("--repeat", type=int, default=1, help="run through the folder this many times")
    return p.parse_args()


def rail_strip(width: int, height: int, offset: int) -> np.ndarray:
    """A dark rail with light tick marks every 80 px; the offset moves the ticks (motion cue)."""
    strip = np.full((height, width, 3), 60, np.uint8)
    for x in range(-80 + offset % 80, width, 80):
        cv2.rectangle(strip, (x, height // 2 - 4), (x + 24, height // 2 + 4), (140, 140, 140), -1)
    return strip


def background(width: int, height: int, offset: int) -> np.ndarray:
    frame = np.full((height, width, 3), 255, np.uint8)
    frame[:LANE_TOP] = rail_strip(width, LANE_TOP, offset)
    frame[LANE_BOTTOM:] = rail_strip(width, height - LANE_BOTTOM, offset)
    return frame


def prepare_part(path: Path, rng: random.Random) -> np.ndarray:
    """A photo as it will sit on the belt: resized, slightly turned (round parts land at
    any angle), white corners so it blends into the backlit belt."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"not an image: {path}")
    img = cv2.resize(img, (BOTTLE_PX, BOTTLE_PX), interpolation=cv2.INTER_AREA)
    angle = rng.uniform(0, 360)
    m = cv2.getRotationMatrix2D((BOTTLE_PX / 2, BOTTLE_PX / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (BOTTLE_PX, BOTTLE_PX), borderValue=(255, 255, 255))


def paste(frame: np.ndarray, part: np.ndarray, x: int, y: int) -> None:
    """Paste a part at (x, y), clipped to the frame. White pixels of the part are the belt."""
    h, w = part.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    frame[y0:y1, x0:x1] = part[y0 - y:y1 - y, x0 - x:x1 - x]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    folder = Path(args.images)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"No images found in {folder}")
    order = []
    for _ in range(args.repeat):
        batch = list(paths)
        rng.shuffle(batch)
        order += batch

    parts = []  # (image path, part image, start x, y)
    lane_mid = (LANE_TOP + LANE_BOTTOM) // 2
    for i, path in enumerate(order):
        parts.append({"path": path, "image": prepare_part(path, rng),
                      "x0": -BOTTLE_PX - i * args.spacing,
                      "y": lane_mid - BOTTLE_PX // 2 + rng.randint(-12, 12)})

    total_travel = WIDTH + BOTTLE_PX + (len(parts) - 1) * args.spacing
    n_frames = total_travel // args.speed + args.fps  # plus one second of empty belt at the end
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise SystemExit("Could not open the video writer (OpenCV without mp4 support?)")

    manifest = []
    sx, sy, sw, sh = STATION
    for part in parts:
        manifest.append({"image": part["path"].name, "label": part["path"].stem.rsplit("_", 1)[0],
                         "enters_station": None, "leaves_station": None})
    for f in range(n_frames):
        frame = background(WIDTH, HEIGHT, offset=f * args.speed)
        for part, entry in zip(parts, manifest):
            x = part["x0"] + f * args.speed
            paste(frame, part["image"], x, part["y"])
            inside = x > sx and x + BOTTLE_PX < sx + sw  # same rule as acquisition.find_parts
            if inside and entry["enters_station"] is None:
                entry["enters_station"] = f
            if inside:
                entry["leaves_station"] = f
        writer.write(frame)
    writer.release()

    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "video": out.name, "fps": args.fps, "frames": n_frames, "width": WIDTH, "height": HEIGHT,
        "station_roi": {"x": sx, "y": sy, "w": sw, "h": sh}, "speed_px_per_frame": args.speed,
        "seed": args.seed, "parts": manifest}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({n_frames} frames, {n_frames / args.fps:.0f} s, {len(parts)} parts) "
          f"and {manifest_path.name}")


if __name__ == "__main__":
    main()
