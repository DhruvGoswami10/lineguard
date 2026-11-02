"""Download the DINOv2 encoder weights that technique 3 (FoundAD) is built on.

    python -m src.download_backbone                 # ViT-S/14, 88 MB (default)
    python -m src.download_backbone --variant vitb14  # ViT-B/14, 346 MB (optional)

Standard library only, like download_data.py. The file is fetched once from Meta's
public server into weights/ and checked against its SHA-256 hash, so every machine
gets byte-identical weights. After this one download, training, evaluation, the demo
and the line runner all work offline. Re-running is safe: a verified file is skipped.

Source and licence: https://github.com/facebookresearch/dinov2 (Apache 2.0).
Reference: Oquab et al. (2024), DINOv2: Learning robust visual features without supervision.
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

BACKBONES = {
    "vits14": {
        "file": "dinov2_vits14_pretrain.pth",
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "sha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
        "bytes": 88_283_115,
    },
    "vitb14": {
        "file": "dinov2_vitb14_pretrain.pth",
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "sha256": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
        "bytes": 346_378_731,
    },
}
DEFAULT_DIR = "weights"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backbone_path(variant: str, folder=DEFAULT_DIR) -> Path:
    return Path(folder) / BACKBONES[variant]["file"]


def is_valid(path: Path, variant: str) -> bool:
    return path.is_file() and sha256_of(path) == BACKBONES[variant]["sha256"]


def fetch(variant: str, folder=DEFAULT_DIR) -> Path:
    """Download the weights unless a verified copy exists. Returns the file path."""
    meta, target = BACKBONES[variant], backbone_path(variant, folder)
    if is_valid(target, variant):
        print(f"{target} already present and verified.")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    print(f"Downloading {meta['url']} ({meta['bytes'] / 1e6:.0f} MB) ...")
    with urllib.request.urlopen(meta["url"], timeout=60) as resp, open(partial, "wb") as f:
        shutil.copyfileobj(resp, f)
    if sha256_of(partial) != meta["sha256"]:
        partial.unlink()
        raise RuntimeError("Downloaded file failed its SHA-256 check; please re-run.")
    partial.replace(target)
    print(f"Saved {target} (SHA-256 verified).")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DINOv2 weights into weights/.")
    parser.add_argument("--variant", default="vits14", choices=sorted(BACKBONES))
    parser.add_argument("--out", default=DEFAULT_DIR, help="folder to save into")
    args = parser.parse_args()
    try:
        fetch(args.variant, args.out)
    except (OSError, RuntimeError) as err:
        sys.exit(f"Download failed: {err}\nCheck your connection and re-run.")


if __name__ == "__main__":
    main()
