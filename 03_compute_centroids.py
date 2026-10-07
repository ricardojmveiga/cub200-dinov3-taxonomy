#!/usr/bin/env python3
"""Step 3 -- per-species centroids + the kNN@1@Order check behind Figure 2 (any device).

From the 4096-D embeddings (same float32, per-image L2 preprocessing as Table 1) it writes:
  * data/cub_species_centroids_4096.npy   (200 x 4096)  re-normalised species centroids
  * data/cub_species_to_{order,family,genus}.npy         species -> rank label
and recomputes kNN@1@Order (each image's nearest neighbour shares its order) -- the
manuscript's 99.9% local-order coherence.

The nearest-neighbour search runs on the auto-detected device: torch cuda/mps (a single
11788x11788 cosine matmul, batched) or a scikit-learn CPU fallback if torch is absent. CPU
uses all-cores-minus-one. The centroid outputs are byte-identical across devices.

    python 03_compute_centroids.py                 # auto device
    python 03_compute_centroids.py --device cpu    # force CPU (cores-1)

It overwrites the bundled centroid files in data/ (identical if the embeddings are the released
ones); use --out DIR to write them elsewhere.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reprocub.common import (DATA, EMBEDDINGS_NPY, configure_cpu_threads, device_label,
                             get_device, load_labels)


def knn1_order(X: np.ndarray, order: np.ndarray, device: str, batch: int = 512) -> tuple[int, int]:
    """Fraction of images whose nearest neighbour (cosine, excl. self) shares its order.
    Returns (correct, N). X is already L2-normalised so cosine == dot."""
    N = len(X)
    try:
        import torch
        if device == "cpu":
            configure_cpu_threads()
        Xt = torch.tensor(X, device=device)
        ordt = torch.tensor(order, device=device)
        correct = 0
        for i in range(0, N, batch):
            sims = Xt[i:i + batch] @ Xt.T
            rows = torch.arange(sims.shape[0], device=device)
            sims[rows, torch.arange(i, i + sims.shape[0], device=device)] = -2.0  # mask self
            nn = sims.argmax(1)
            correct += int((ordt[nn] == ordt[i:i + sims.shape[0]]).sum())
        return correct, N
    except ImportError:
        configure_cpu_threads()
        from sklearn.neighbors import NearestNeighbors
        _, ind = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X).kneighbors(X)
        rows = np.arange(N)
        nn = np.where(ind[:, 0] == rows, ind[:, 1], ind[:, 0])   # drop self (a 0-distance
        return int((order[nn] == order).sum()), N               # tie can put self in col 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, default=EMBEDDINGS_NPY,
                    help="embeddings .npy (default: data/cub200_cls_embeddings.npy)")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None,
                    help="device for the nearest-neighbour search (default: auto)")
    ap.add_argument("--out", type=Path, default=DATA,
                    help="folder for the centroid files (default: data/, overwriting the bundled ones)")
    args = ap.parse_args()

    if not args.emb.exists():
        print(f"  embeddings not found at {args.emb} -- run steps 0-1 or use the bundled "
              f"centroids (Figure 2 regenerates from data/cub_species_centroids_4096.npy)", flush=True)
        return 1

    dev = get_device(args.device)
    print(f"[centroids] device={device_label(dev)}", flush=True)
    labels, _, _ = load_labels()
    order, family, genus, species = (labels[r] for r in ("order", "family", "genus", "species"))

    X = np.load(args.emb).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)          # per-image L2 (as in Table 1)
    sp_ids = sorted(set(species.tolist()))

    cent = np.stack([X[species == s].mean(0) for s in sp_ids]).astype(np.float32)
    cent /= (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12)     # re-normalise centroids
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "cub_species_centroids_4096.npy", cent)
    for rank, arr in (("order", order), ("family", family), ("genus", genus)):
        np.save(args.out / f"cub_species_to_{rank}.npy",
                np.array([arr[species == s][0] for s in sp_ids]))
    print(f"  wrote centroids {cent.shape} + label vectors", flush=True)

    t0 = time.perf_counter()
    correct, N = knn1_order(X, order, dev)
    print(f"  kNN@1@Order = {100 * correct / N:.1f}%  ({correct}/{N})  "
          f"[{time.perf_counter()-t0:.3f}s on {dev}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
