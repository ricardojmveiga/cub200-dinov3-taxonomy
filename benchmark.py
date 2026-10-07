#!/usr/bin/env python3
"""Profile the pipeline's hot paths across devices and record the timings.

Two representative workloads:
  * knn      -- the 11788x11788 cosine nearest-neighbour search behind kNN@1@Order
               (a big matmul; runs on CUDA / Apple-MPS / CPU via torch).
  * cluster  -- k-means at K=200 on the 11788x4096 features. A fixed-budget Lloyd
               (matmul assignment, 25 iters) runs as the SAME algorithm on every device
               (CUDA / MPS / CPU) so the GPU cells are comparable; scikit-learn's MiniBatch
               k-means (the paper's actual clusterer) is timed on CPU only as a reference.

By default it benchmarks every device available on the machine (CUDA, MPS, CPU-all-cores-minus-one)
and writes one JSON per run to benchmarks/. Uses the real embeddings if present, otherwise a
synthetic matrix of the SAME shape (device timing is representative either way).

    python benchmark.py --label my-laptop  # all available devices; names the output file
    python benchmark.py --device cuda      # just CUDA
    python benchmark.py --reps 3           # quicker: median of 3 timed reps (default 5)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reprocub.common import (BENCHMARKS, EMBEDDINGS_NPY, configure_cpu_threads, device_label,
                             env_summary, n_threads)


def _features() -> tuple[np.ndarray, str]:
    if EMBEDDINGS_NPY.exists():
        X = np.load(EMBEDDINGS_NPY).astype(np.float32)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return X, "real"
    rng = np.random.default_rng(0)
    X = rng.standard_normal((11788, 4096)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True), "synthetic"


def _available_devices(force: str | None):
    if force:
        return [force]
    devs = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            devs.insert(0, "cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devs.insert(0, "mps")
    except Exception:
        pass
    return devs


def _sync(device: str) -> None:
    """Drain the async device queue so the clock stops only after work completes.
    (The earlier version synced CUDA only, so MPS timings were bogus submit-times.)"""
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and getattr(torch, "mps", None) is not None:
        torch.mps.synchronize()
    # cpu: eager ops already block the calling thread


def bench_knn(X: np.ndarray, device: str, reps: int, batch: int = 512) -> float:
    """Median seconds for the full 11788x11788 cosine kNN (excl. host<->device copy of X)."""
    import torch
    if device == "cpu":
        configure_cpu_threads()
    Xt = torch.tensor(X, device=device)

    def once() -> float:
        _sync(device)
        t0 = time.perf_counter()
        for i in range(0, len(X), batch):
            sims = Xt[i:i + batch] @ Xt.T
            sims[torch.arange(sims.shape[0]), torch.arange(i, i + sims.shape[0])] = -2.0
            sims.argmax(1)
        _sync(device)
        return time.perf_counter() - t0

    once()                                            # warm-up (JIT / cudnn / allocator)
    return statistics.median(once() for _ in range(reps))


def bench_cluster(X: np.ndarray, reps: int) -> float:
    """Median seconds for scikit-learn MiniBatch k-means at K=200 on CPU (cores-1).
    This is the paper's actual clusterer (a *different*, mini-batch algorithm) -- kept as
    a CPU-only reference, NOT the same as the device-portable torch k-means below."""
    configure_cpu_threads()
    from sklearn.cluster import MiniBatchKMeans

    def once() -> float:
        t0 = time.perf_counter()
        MiniBatchKMeans(n_clusters=200, random_state=42, n_init=10, batch_size=1024).fit_predict(X)
        return time.perf_counter() - t0

    return statistics.median(once() for _ in range(reps))


def bench_cluster_torch(X: np.ndarray, device: str, reps: int, K: int = 200,
                        iters: int = 25, seed: int = 42) -> float:
    """Median seconds for a fixed-budget Lloyd k-means (K=200) as the SAME algorithm on
    cuda / mps / cpu -- fills the GPU cells sklearn's CPU-only MiniBatchKMeans cannot.

    Rows of X are unit-L2, so nearest-Euclidean ranking is a single (N,K) GEMM plus an O(K)
    centroid-norm term (||x-c||^2 = 1 - 2<x,c> + ||c||^2 => argmin == argmax(<x,c> - .5||c||^2)),
    routed through cuBLAS / MPS-BLAS / MKL instead of torch.cdist -- the SAME objective, faster.
    Fixed iters (no tol early-stop) so every device does identical FLOPs; timing excludes the
    host->device copy; warm-up + device-correct sync; seed-42 CPU-generator init (identical C0
    on every device); empty clusters reseeded to the worst-served rows (sklearn's strategy)."""
    import torch
    if device == "cpu":
        configure_cpu_threads()
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)   # MPS: fp32 only; copy once, untimed
    N, D = Xt.shape
    ones = torch.ones(N, device=device, dtype=Xt.dtype)
    chunk = max(1024, min(N, int(256e6 // (K * 4))))              # bound the (chunk,K) fp32 buffer

    def lloyd() -> None:
        g = torch.Generator().manual_seed(seed)                  # CPU gen (MPS gen unsupported)
        C = Xt[torch.randperm(N, generator=g)[:K].to(device)].clone()
        for _ in range(iters):
            c_bias = 0.5 * (C * C).sum(1)                        # (K,) -> exact Euclidean ranking
            labels = torch.empty(N, dtype=torch.long, device=device)
            best = torch.empty(N, dtype=Xt.dtype, device=device)
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                scores = (Xt[s:e] @ C.T).sub_(c_bias)            # (chunk,K) via BLAS, in-place bias
                best[s:e], labels[s:e] = scores.max(1)
            csum = torch.zeros(K, D, device=device, dtype=Xt.dtype).index_add_(0, labels, Xt)
            cnt = torch.zeros(K, device=device, dtype=Xt.dtype).index_add_(0, labels, ones)
            empty = cnt == 0
            C = csum / cnt.clamp_min(1.0).unsqueeze(1)
            n_empty = int(empty.sum())                           # one cheap sync/iter
            if n_empty:                                          # reseed to farthest (smallest best) rows
                C[empty] = Xt[torch.topk(best, n_empty, largest=False).indices]
        _sync(device)

    lloyd()                                                      # warm-up (allocator / BLAS / power-state)

    def once() -> float:
        _sync(device)
        t0 = time.perf_counter()
        lloyd()
        return time.perf_counter() - t0

    return statistics.median(once() for _ in range(reps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None,
                    help="time only this device (default: every device available)")
    ap.add_argument("--ops", nargs="*", default=["knn", "cluster"], choices=["knn", "cluster"],
                    help="which hot paths to time")
    ap.add_argument("--reps", type=int, default=5, help="repetitions per timing (the median is kept)")
    ap.add_argument("--label", default="local",
                    help="machine name for the output file (the hostname is never recorded)")
    args = ap.parse_args()

    X, source = _features()
    env = env_summary()
    print(f"[bench] features={X.shape} ({source})  machine={args.label}  {env['os']}/{env['machine']}", flush=True)

    results = {"machine": args.label, "env": env, "features": source, "reps": args.reps,
               "timings": []}
    for dev in _available_devices(args.device):
        rec = {"device": dev, "label": device_label(dev), "threads": n_threads() if dev == "cpu" else None}
        try:
            if "knn" in args.ops:
                rec["knn_s"] = round(bench_knn(X, dev, args.reps), 4)
            if "cluster" in args.ops:
                rec["cluster_torch_s"] = round(bench_cluster_torch(X, dev, args.reps), 4)  # every device
                if dev == "cpu":                            # sklearn (paper's algo) is CPU-only
                    rec["cluster_s"] = round(bench_cluster(X, args.reps), 4)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        results["timings"].append(rec)
        msg = "  ".join(f"{k}={v}" for k, v in rec.items() if k not in ("label",))
        print(f"[bench] {rec['label']}: {msg}", flush=True)

    BENCHMARKS.mkdir(parents=True, exist_ok=True)
    tag = f"{args.label}_{(args.device or 'all')}"
    dst = BENCHMARKS / f"benchmark_{tag}.json"
    dst.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[bench] wrote {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
