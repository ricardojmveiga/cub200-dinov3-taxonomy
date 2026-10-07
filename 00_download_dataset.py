#!/usr/bin/env python3
"""Step 0 -- download the CUB-200-2011 images (any OS).

Pure-Python: :mod:`urllib` fetches the ~1.1 GB tarball and :mod:`tarfile` extracts it,
so there is no dependency on ``wget``/``curl``/``tar`` and it runs identically on
Windows, macOS and Linux. The download resumes if interrupted and the extraction is
verified against the bundled image order (data/cub200_image_paths.json).

    python 00_download_dataset.py            # download + extract into ./dataset
    python 00_download_dataset.py --check     # only verify an existing extraction
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reprocub.common import DATASET_DIR, DATA

# Caltech DOI-hosted mirror of the official CUB-200-2011 release.
URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
# CaltechDATA answers 403 Forbidden to urllib's default User-Agent, so the client names itself.
HEADERS = {"User-Agent": "cub200-dinov3-taxonomy (+https://github.com/ricardojmveiga/cub200-dinov3-taxonomy)"}
TGZ = DATASET_DIR / "CUB_200_2011.tgz"
TGZ_BYTES = 1_150_585_339                                   # size of the CaltechDATA tarball
IMAGES_DIR = DATASET_DIR / "CUB_200_2011" / "images"


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    have = dst.stat().st_size if dst.exists() else 0
    req = urllib.request.Request(url, headers={**HEADERS, **({"Range": f"bytes={have}-"} if have else {})})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:                                   # already complete
            print("  already downloaded", flush=True); return
        raise
    total = have + int(resp.headers.get("Content-Length", 0))
    mode = "ab" if (have and resp.status == 206) else "wb"
    done = have if mode == "ab" else 0
    print(f"  downloading -> {dst.name} ({total / 1e9:.2f} GB){' [resume]' if mode=='ab' else ''}", flush=True)
    with open(dst, mode) as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk); done += len(chunk)
            if total:
                pct = 100 * done / total
                print(f"\r  {done/1e9:5.2f}/{total/1e9:.2f} GB  {pct:5.1f}%", end="", flush=True)
    print(flush=True)


def _extract(tgz: Path, dst: Path) -> None:
    print(f"  extracting {tgz.name} ...", flush=True)
    with tarfile.open(tgz, "r:gz") as tar:
        # path-traversal guard (safe on every OS)
        base = dst.resolve()
        for m in tar.getmembers():
            if not (base / m.name).resolve().is_relative_to(base):
                raise RuntimeError(f"unsafe path in tar: {m.name}")
        # Py3.12+ 'data' filter also blocks unsafe symlink/hardlink/device members
        # (PEP-706); keep the manual name guard above as the pre-3.12 fallback.
        if sys.version_info >= (3, 12):
            tar.extractall(dst, filter="data")
        else:
            tar.extractall(dst)


def verify() -> bool:
    paths = json.loads((DATA / "cub200_image_paths.json").read_text(encoding="utf-8"))
    missing = [p for p in paths[:50] if not (IMAGES_DIR / p).exists()]  # spot-check 50
    n_ok = sum((IMAGES_DIR / p).exists() for p in paths)
    print(f"  images present: {n_ok}/{len(paths)}", flush=True)
    if missing:
        print(f"  MISSING e.g. {missing[0]}", flush=True)
    return n_ok == len(paths)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify an existing extraction only")
    args = ap.parse_args()
    if not args.check:
        if not IMAGES_DIR.exists():
            if not TGZ.exists() or TGZ.stat().st_size < TGZ_BYTES:   # missing or partial: (re)start
                _download(URL, TGZ)
            _extract(TGZ, DATASET_DIR)
        else:
            print("  dataset already extracted", flush=True)
    ok = verify()
    print("OK" if ok else "INCOMPLETE -- re-run without --check", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
