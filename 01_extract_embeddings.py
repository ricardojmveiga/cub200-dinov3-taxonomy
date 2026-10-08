#!/usr/bin/env python3
"""Step 1 -- extract frozen DINOv3-7B CLS embeddings for CUB-200 (any OS, GPU strongly advised).

Reproduces data/cub200_cls_embeddings.npy = (11788, 4096) float32, L2-normalised, in the
exact order of data/cub200_image_paths.json (so it lines up with the bundled labels).

Faithful to the extraction behind the paper's embeddings:
  * model  facebook/dinov3-vit7b16-pretrain-lvd1689m
  * input  512x512 (multiple of patch_size 16)
  * token  CLS = last_hidden_state[:, 0, :]  (index 0, before the 4 register tokens)
  * dtype  bfloat16 weights, float32 pixels, forward pass under bfloat16 autocast (the default on
           CUDA and Apple MPS); float32 throughout on CPU. Cast to float32 BEFORE L2-normalising.

The checkpoint is gated: accept the DINOv3 License on
https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m and log in once
(``hf auth login``, or set HF_TOKEN) before the first run. It downloads ~27 GB on first use.

Device is auto-detected (CUDA -> MPS -> CPU). NOTE: this is a 7-billion-parameter model: about
13.5 GB of weights in bfloat16, 27 GB in float32. A CUDA GPU is fastest. On a CPU, keep the default
float32: on a 24-core x86 machine it took about 17 s per image (28 GB peak resident), while
--dtype bf16 took 97 s per image (40 GB). Without a suitable machine, use the released embeddings
(fetch_embeddings.py); steps 2-4 need no GPU.

    python 01_extract_embeddings.py                 # auto device, into data/
    python 01_extract_embeddings.py --limit 10 --out data/smoke_10.npy   # quick check (first 10 images)

Without --out it writes data/cub200_cls_embeddings.npy, replacing any fetched copy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reprocub.common import DATA, DATASET_DIR, EMBEDDINGS_NPY, get_device, device_label

MODEL_ID = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
INPUT_SIZE, EXPECTED_DIM = 512, 4096
IMAGES_DIR = DATASET_DIR / "CUB_200_2011" / "images"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None,
                    help="compute device (default: auto, CUDA -> MPS -> CPU)")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto",
                    help="weights/compute precision (default: bf16 on CUDA and MPS, fp32 on CPU, where bf16 is slower)")
    ap.add_argument("--batch-size", type=int, default=8, help="images per forward pass")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke)")
    ap.add_argument("--out", type=Path, default=EMBEDDINGS_NPY,
                    help="output .npy (default: data/cub200_cls_embeddings.npy)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    dev = get_device(args.device)
    print(f"[extract] device={device_label(dev)}", flush=True)
    paths = json.loads((DATA / "cub200_image_paths.json").read_text(encoding="utf-8"))
    if args.limit:
        paths = paths[:args.limit]
    if not IMAGES_DIR.exists():
        print(f"  dataset not found at {IMAGES_DIR} -- run 00_download_dataset.py first", flush=True)
        return 1

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    use_bf16 = args.dtype == "bf16" or (args.dtype == "auto" and dev in ("cuda", "mps"))
    print(f"[extract] precision={'bfloat16' if use_bf16 else 'float32'}", flush=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if use_bf16 else torch.float32).to(dev).eval()

    out = np.empty((len(paths), EXPECTED_DIM), dtype=np.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(paths), args.batch_size):
            batch = paths[i:i + args.batch_size]
            imgs = [Image.open(IMAGES_DIR / p).convert("RGB") for p in batch]
            px = proc(imgs, size={"height": INPUT_SIZE, "width": INPUT_SIZE},
                      return_tensors="pt")["pixel_values"].to(dev)   # float32 pixels
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=use_bf16):
                cls = model(pixel_values=px).last_hidden_state[:, 0, :]  # CLS token
            cls = cls.float()                                           # bf16 -> f32 before norm
            cls = torch.nn.functional.normalize(cls, dim=1)
            out[i:i + len(batch)] = cls.cpu().numpy()
            print(f"\r  {min(i + len(batch), len(paths))}/{len(paths)}", end="", flush=True)
    print(f"\n[extract] {len(paths)} imgs in {time.perf_counter()-t0:.1f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    print(f"  wrote {args.out}  shape={out.shape} dtype={out.dtype}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
