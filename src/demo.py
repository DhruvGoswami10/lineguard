"""LineGuard demo: inspect bottle images, print OK / REJECT, save a heatmap figure.

    python -m src.demo --model ae  --input samples/
    python -m src.demo --model cnn --input samples/broken_small_010.png

Needs only weights/ and the images: no dataset, no training, no internet.
For each image it prints the score, the threshold and the verdict, and saves an
image | heatmap | verdict figure to results/demo/<model>_<image name>.png.
"""
import argparse
import sys
from pathlib import Path

from src.common import get_device, set_seed
from src.data import load_image
from src.inference import inspect, load_model
from src.visualize import save_verdict_figure

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
MODEL_NAMES = {"ae": "Convolutional autoencoder", "cnn": "ResNet-18 classifier"}


def find_images(path: Path) -> list[Path]:
    """A single image file, or every image in a folder (sorted by name)."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"Input not found: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LineGuard demo: OK / REJECT with a heatmap per image.")
    p.add_argument("--model", required=True, choices=["ae", "cnn"],
                   help="ae = autoencoder (technique 1), cnn = ResNet-18 (technique 2)")
    p.add_argument("--input", required=True, help="an image file or a folder of images")
    p.add_argument("--weights", default="weights", help="folder containing ae.pt and cnn.pt")
    p.add_argument("--out", default="results/demo", help="where the figures are saved")
    p.add_argument("--device", default="auto", help="auto (GPU if available), cpu or cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    device = get_device(args.device)
    try:
        images = find_images(Path(args.input))
        model, ckpt = load_model(Path(args.weights) / f"{args.model}.pt", device)
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}")
    if not images:
        sys.exit(f"Error: no images ({', '.join(sorted(IMAGE_SUFFIXES))}) found in {args.input}")

    fmt = ".5g" if args.model == "ae" else ".5f"  # probabilities: fixed decimals (0.999996 -> 1.00000)
    print(f"LineGuard demo | {MODEL_NAMES[args.model]} | REJECT if score >= {ckpt['threshold']:{fmt}} | {device}")
    print(f"{'image':<28}{'score':>12}{'threshold':>12}  verdict")
    counts = {"OK": 0, "REJECT": 0}
    for path in images:
        try:
            image = load_image(path)
        except OSError as err:  # unreadable or not really an image: skip it, keep going
            print(f"{path.name:<28}  skipped ({err})")
            continue
        result = inspect(model, ckpt, image, device, with_heatmap=True)
        counts[result["verdict"]] += 1
        print(f"{path.name:<28}{result['score']:>12{fmt}}{result['threshold']:>12{fmt}}  {result['verdict']}")
        save_verdict_figure(image, result, args.model, Path(args.out) / f"{args.model}_{path.stem}.png",
                            f"LineGuard | {MODEL_NAMES[args.model]} | {path.name}")
    print(f"\n{sum(counts.values())} images: {counts['OK']} OK, {counts['REJECT']} REJECT. "
          f"Figures saved to {args.out}/")


if __name__ == "__main__":
    main()
