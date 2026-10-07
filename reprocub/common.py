"""Cross-platform helpers shared by every step of the reproduction pipeline.

Everything here is OS-agnostic: paths use :mod:`pathlib` (so Windows back-slashes,
macOS and Linux forward-slashes all work), the compute device is auto-detected
(CUDA -> Apple-MPS -> CPU), and CPU runs use *all cores minus one* by default so a
laptop stays responsive. No absolute paths, no shell-isms, no platform branches in
the callers -- import from here instead.
"""
from __future__ import annotations

import json
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths (pathlib)
# reprocub/common.py  ->  parents[1] is the repository root
ROOT: Path = Path(__file__).resolve().parents[1]
DATA: Path = ROOT / "data"
FIGURES: Path = ROOT / "figures"
EXPERIMENTS_OUT: Path = DATA / "experiments_out"
BENCHMARKS: Path = ROOT / "benchmarks"
DATASET_DIR: Path = ROOT / "dataset"                      # CUB-200-2011 lands here
EMBEDDINGS_NPY: Path = DATA / "cub200_cls_embeddings.npy"  # 11788x4096, not shipped (193 MB)
LABELS_JSON: Path = DATA / "cub200_labels.json"

for _d in (DATA, FIGURES, EXPERIMENTS_OUT, BENCHMARKS):
    _d.mkdir(parents=True, exist_ok=True)

RANKS = ("order", "family", "genus", "species")


# --------------------------------------------------------------- CPU thread count
def n_threads() -> int:
    """All physical/logical cores minus one (>=1) -- the project's CPU policy."""
    return max(1, (os.cpu_count() or 2) - 1)


def configure_cpu_threads(n: int | None = None) -> int:
    """Pin BLAS/OpenMP/torch to ``n`` threads (default: cores-1). Call before heavy
    numpy/torch work.

    The env vars only bind if they are set *before* the BLAS pool spins up (i.e. before
    numpy is imported), so on their own they are a no-op once numpy is already loaded.
    We therefore also call :mod:`threadpoolctl` (a scikit-learn dependency, always present)
    to constrain the *already-initialised* OpenBLAS/MKL pools at run time, plus
    :func:`torch.set_num_threads`. This makes the "cores-1" policy actually effective."""
    n = n_threads() if n is None else max(1, n)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)
    try:                                     # runtime-effective even after numpy/sklearn import
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass
    return n


# ------------------------------------------------------------------ device pick
def get_device(prefer: str | None = None) -> str:
    """Return the best available torch device string: 'cuda' > 'mps' (Apple) > 'cpu'.
    ``prefer`` forces one ('cuda'/'mps'/'cpu'); falls back with a note if unavailable.
    Returns 'cpu' (never raises) when torch is absent."""
    try:
        import torch
    except Exception:
        return "cpu"
    avail = {"cpu": True,
             "cuda": torch.cuda.is_available(),
             "mps": getattr(torch.backends, "mps", None) is not None
                    and torch.backends.mps.is_available()}
    if prefer:
        if avail.get(prefer):
            return prefer
        print(f"[device] '{prefer}' unavailable; falling back", flush=True)
    for d in ("cuda", "mps", "cpu"):
        if avail[d]:
            return d
    return "cpu"


def device_label(dev: str) -> str:
    """Human-readable device name for logs/benchmarks."""
    try:
        import torch
        if dev == "cuda" and torch.cuda.is_available():
            return f"CUDA:{torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    if dev == "mps":
        return f"MPS:AppleSilicon ({platform.processor() or platform.machine()})"
    return f"CPU:{platform.processor() or platform.machine()} x{n_threads()} threads"


# ------------------------------------------------------------------------ labels
def load_labels(path: Path | None = None):
    """(labels dict rank->int array, oracle_k dict, n) from the de-identified JSON."""
    path = Path(path) if path else LABELS_JSON
    d = json.loads(path.read_text(encoding="utf-8"))
    labels = {r: np.asarray(d["labels"][r], dtype=int) for r in d["ranks"]}
    oracle = d["oracle_k"]
    for r in d["ranks"]:
        assert len(set(labels[r].tolist())) == oracle[r], f"{r}: {oracle[r]}"
    return labels, oracle, int(d["n"])


# ------------------------------------------------------------------------ timing
@contextmanager
def timed(name: str, sink: dict | None = None):
    """Context manager that prints and (optionally) records wall-clock seconds."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[time] {name}: {dt:.3f} s", flush=True)
    if sink is not None:
        sink[name] = round(dt, 4)


def env_summary() -> dict:
    """Reproducibility fingerprint: OS, python, numpy/scipy/scikit-learn, and (if present) torch/CUDA."""
    info = {"os": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "python": platform.python_version(),
            "numpy": np.__version__, "cpu_count": os.cpu_count(), "threads": n_threads()}
    for mod, key in (("scipy", "scipy"), ("sklearn", "scikit_learn")):
        try:
            info[key] = __import__(mod).__version__
        except Exception:
            info[key] = None
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = bool(getattr(torch.backends, "mps", None)
                                     and torch.backends.mps.is_available())
    except Exception:
        info["torch"] = None
    return info
