#!/usr/bin/env python3
"""Fetch the pre-computed DINOv3-7B CLS embeddings (193 MB) so the pipeline runs GPU-free.

Two hosted copies of the same file (no login needed):
  * Hugging Face dataset  ricardojmveiga/cub200-dinov3-7b-embeddings
  * GitHub release asset  ricardojmveiga/cub200-dinov3-taxonomy v1.0.0

    python fetch_embeddings.py                  # auto: Hugging Face first, the GitHub release if that fails
    python fetch_embeddings.py --source hf      # only the Hugging Face dataset
    python fetch_embeddings.py --source github  # only the GitHub release asset

The download is checked against the published SHA-256 before it is saved as
data/cub200_cls_embeddings.npy. The Hugging Face route needs ``huggingface_hub`` (installed with
requirements-torch.txt); without it, auto mode goes straight to the GitHub release.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reprocub.common import EMBEDDINGS_NPY

HF_REPO = "ricardojmveiga/cub200-dinov3-7b-embeddings"      # Hugging Face dataset
GH_REPO = "ricardojmveiga/cub200-dinov3-taxonomy"           # GitHub repo (release host)
TAG = "v1.0.0"
ASSET = "cub200_cls_embeddings.npy"
GH_URL = f"https://github.com/{GH_REPO}/releases/download/{TAG}/{ASSET}"
SHA256 = "af63ce50ee4644f49d0af8b2b3f7b3fd3860f48c21bff25d3b0e99bd39b21417"
SHAPE = (11788, 4096)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _from_hf(part: Path) -> None:
    from huggingface_hub import hf_hub_download             # ImportError -> caller falls back
    print(f"  Hugging Face: {HF_REPO} / {ASSET} ...", flush=True)
    cached = hf_hub_download(HF_REPO, ASSET, repo_type="dataset")
    shutil.copyfile(cached, part)                           # copy out of the HF cache


def _from_github(part: Path) -> None:
    print(f"  GitHub release: {GH_URL} ...", flush=True)
    try:
        with urllib.request.urlopen(GH_URL, timeout=60) as r, open(part, "wb") as f:
            total, done = int(r.headers.get("Content-Length", 0)), 0
            while chunk := r.read(1 << 20):
                f.write(chunk); done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:6.1f}/{total / 1e6:.1f} MB", end="", flush=True)
        print(flush=True)
    except Exception as e:
        if not shutil.which("gh"):
            raise
        # a logged-in GitHub CLI can still reach the asset when plain HTTPS cannot (e.g. a private fork)
        print(f"  plain download failed ({type(e).__name__}); trying the GitHub CLI ...", flush=True)
        part.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["gh", "release", "download", TAG, "--repo", GH_REPO,
                            "--pattern", ASSET, "--dir", tmp], check=True)
            shutil.move(str(Path(tmp) / ASSET), part)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download the released DINOv3-7B CUB-200 embeddings.")
    ap.add_argument("--source", choices=["auto", "hf", "github"], default="auto",
                    help="where to download from (default: auto, Hugging Face then GitHub)")
    ap.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = ap.parse_args()

    import numpy as np

    dst = EMBEDDINGS_NPY
    if dst.exists() and not args.force:
        shape = np.load(dst, mmap_mode="r").shape            # cheap: mmap, no full read
        same = _sha256(dst) == SHA256
        print(f"  already present: {dst}  shape={shape}  "
              f"({'the released file' if same else 'NOT the released file; --force replaces it'})", flush=True)
        if shape != SHAPE:
            print(f"  expected shape {SHAPE}; delete it or rerun with --force", flush=True)
            return 1
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)

    order = {"auto": [("hf", _from_hf), ("github", _from_github)],
             "hf": [("hf", _from_hf)], "github": [("github", _from_github)]}[args.source]
    part = dst.with_name(dst.name + ".part")
    for i, (name, fetch) in enumerate(order):
        try:
            fetch(part)
            got = _sha256(part)
            if got != SHA256:
                raise ValueError(f"SHA-256 mismatch: got {got}, expected {SHA256}")
        except Exception as e:
            part.unlink(missing_ok=True)
            if i == len(order) - 1:
                print(f"  {name} failed: {type(e).__name__}: {e}", flush=True)
                return 1
            if isinstance(e, ImportError):
                print("  huggingface_hub is not installed; using the GitHub release", flush=True)
            else:
                print(f"  {name} failed ({type(e).__name__}: {str(e)[:120]}); trying the next source", flush=True)
            continue
        part.replace(dst)
        X = np.load(dst, mmap_mode="r")
        print(f"  saved {dst}  shape={X.shape} dtype={X.dtype}  SHA-256 OK  (via {name})", flush=True)
        return 0 if X.shape == SHAPE else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
