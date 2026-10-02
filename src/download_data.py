"""Download the MVTec AD 'bottle' category into data/mvtec/bottle/.

Downloading uses only the Python standard library. Files come from a Hugging
Face mirror of MVTec AD, pinned to one commit so the data can never change
underneath us, and every image is checked against its SHA-256 hash.

Official source (licence must be accepted on the website):
    https://www.mvtec.com/company/research/datasets/mvtec-ad
Licence: CC BY-NC-SA 4.0 (non-commercial use, attribution, share-alike).

Usage:
    python -m src.download_data
Re-running is safe: files that are already present and verified are skipped.
"""
import argparse
import hashlib
import http.client
import json
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "foersben/mvtec-ad"
REVISION = "c75b39616f84db43677bcc8228caaafaf5096d7f"  # pinned mirror commit
CATEGORY = "bottle"
LIST_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{CATEGORY}?recursive=true"
FILE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{{path}}"
RETRIES = 5


def list_remote_files() -> list[dict]:
    """Return the mirror's metadata (path, size, SHA-256) for every file in the category."""
    files, url = [], LIST_URL
    while url:  # the listing is paginated; follow rel="next" links until the end
        with urllib.request.urlopen(url, timeout=60) as resp:
            files += [item for item in json.load(resp) if item["type"] == "file"]
            url = _next_page(resp.headers.get("Link"))
    return files


def _next_page(link_header: str | None) -> str | None:
    if link_header and 'rel="next"' in link_header:
        return link_header.split(";")[0].strip("<> ")
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path, meta: dict) -> bool:
    """True if the local file matches the mirror (hash for images, size for text files)."""
    if not path.is_file():
        return False
    if "lfs" in meta:
        return sha256_of(path) == meta["lfs"]["oid"]
    return path.stat().st_size == meta["size"]


def fetch(meta: dict, out_root: Path) -> bool:
    """Download one file unless a verified copy exists. Returns True if downloaded.

    Connections sometimes drop mid-file, so each file gets a few attempts and
    is only moved into place once its hash checks out.
    """
    target = out_root / meta["path"]
    if is_valid(target, meta):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")  # never leave a half-written image behind
    problem = ""
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(FILE_URL.format(path=meta["path"]), timeout=60) as resp, \
                    open(partial, "wb") as f:
                shutil.copyfileobj(resp, f)
            if is_valid(partial, meta):
                partial.replace(target)
                return True
            problem = "checksum mismatch"
        except (OSError, http.client.HTTPException) as err:  # dropped connection, timeout, ...
            problem = str(err)
        partial.unlink(missing_ok=True)
        time.sleep(attempt)  # back off a little before retrying
    raise RuntimeError(f"Could not download {meta['path']} after {RETRIES} tries ({problem}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MVTec AD (bottle) into data/mvtec/.")
    parser.add_argument("--out", default="data/mvtec", help="folder that will contain bottle/")
    args = parser.parse_args()
    out_root = Path(args.out)

    try:
        files = list_remote_files()
        print(f"Mirror lists {len(files)} files for '{CATEGORY}'. Downloading to {out_root / CATEGORY} ...")
        with ThreadPoolExecutor(max_workers=4) as pool:  # a few in parallel: ~350 small files
            new = sum(pool.map(lambda meta: fetch(meta, out_root), files))
    except (OSError, RuntimeError, http.client.HTTPException) as err:
        sys.exit(f"Download failed: {err}\nCheck your connection and re-run; finished files are kept.")
    print(f"Done: {new} downloaded, {len(files) - new} already present.")

    from src.data import check_layout  # imported here: needs the packages in requirements.txt
    check_layout(out_root / CATEGORY)


if __name__ == "__main__":
    main()
