#!/usr/bin/env python3
"""Step 2 -- the clustering experiments behind Table 1 and Figure 1 (CPU, any OS).

Three studies on the frozen CLS embeddings:
  E1 divisive  : top-down bisecting k-means vs flat k-means
  E2 ablation  : six standard clusterers per rank (-> Table 1)
  E3 PCA sweep : recovery vs PCA dimensionality (-> Figure 1)

Records both NMI and the chance-corrected ARI. The built-in sanity check: ``--smoke`` runs on
a random matrix, where ARI must be ~0 while NMI is inflated (Vinh 2010). Pure scikit-learn on
CPU using all-cores-minus-one; no GPU. Writes data/experiments_out/*.json. (The reusable
clustering core lives in reprocub/experiments.py so the tests can exercise it directly.)

    python 02_run_experiments.py                    # real run (needs the embeddings)
    python 02_run_experiments.py --smoke            # fast falsification gate (n=500, K capped at 20)
    python 02_run_experiments.py --control          # random-feature control at the real K

A real run writes data/experiments_out/cub_experiments_real.json, which step 4 then uses in place
of the bundled results (E1+E3 took about 25 min and E2 about 53 min on the authors' machine).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reprocub.common import (EMBEDDINGS_NPY, EXPERIMENTS_OUT, RANKS, configure_cpu_threads,
                             env_summary, load_labels)
from reprocub.experiments import SEEDS, control, exp1, exp2, exp3, load_embeddings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--emb", type=Path, default=EMBEDDINGS_NPY,
                    help="embeddings .npy (default: data/cub200_cls_embeddings.npy)")
    ap.add_argument("--smoke", action="store_true", help="random matrix, no embeddings")
    ap.add_argument("--control", action="store_true",
                    help="random-feature control at the REAL oracle K -> cub_experiments_control_rerun.json")
    ap.add_argument("--n", type=int, default=0, help="subsample to N points (smoke defaults to 500)")
    ap.add_argument("--only", nargs="*", choices=["e1", "e2", "e3"], help="run only these studies")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default cores-1; the paper's runs used all 24 cores, and spectral clustering's values depend on it)")
    args = ap.parse_args()
    if args.smoke and not args.n:
        args.n = 500          # spectral/GMM are superlinear in n; keep the bare --smoke gate fast

    t = configure_cpu_threads(args.threads)
    print(f"[experiments] {t} CPU threads (of {os.cpu_count()})", flush=True)
    labels, oracle, n = load_labels()

    if args.control:                                        # random-feature control at the real oracle K
        print(f"[control] random N(0,1) features (n={n}, 4096-D, L2), flat k-means at oracle "
              f"K={oracle}, {len(SEEDS)} seeds", flush=True)
        res = control(labels, oracle)
        out = {"mode": "control", "n": n, "oracle_k": oracle, "seeds": SEEDS,
               "note": "random N(0,1) features, L2-normalised, flat MiniBatch k-means at the REAL oracle K",
               "per_rank": res, "env": env_summary()}
        dst = EXPERIMENTS_OUT / "cub_experiments_control_rerun.json"   # the bundled control stays the reference
        dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
        for r in RANKS:
            print(f"  {r:8s} ARI={res[r]['ari']['mean']:+.4f} (max|{max(abs(v) for v in res[r]['ari']['runs']):.4f}|)"
                  f"  NMI={res[r]['nmi']['mean']:.4f}", flush=True)
        print(f"  wrote {dst}", flush=True)
        return 0

    X = load_embeddings(args.emb, n, args.smoke)
    if args.n and args.n < n:
        idx = np.random.default_rng(0).permutation(n)[:args.n]
        X = X[idx]; labels = {r: labels[r][idx] for r in RANKS}
        oracle = {r: int(len(set(labels[r]))) for r in RANKS}
    if args.smoke:
        # cap K: GMM/bisecting/spectral are slow when K approaches n, and the ARI~0 falsification
        # gate holds at ANY K on noise. (A real run uses the true oracle K.)
        oracle = {r: min(oracle[r], 20) for r in RANKS}
    print(f"  embeddings {X.shape}  {'[SMOKE]' if args.smoke else args.emb.name}", flush=True)

    # bare --smoke is a fast falsification gate: E2 alone shows ARI~0 across all six clusterers
    # (with visible per-method progress); E1/E3 wiring is covered by test_experiments.py.
    run = args.only or (["e2"] if args.smoke else ["e1", "e2", "e3"])
    out = {"mode": "smoke" if args.smoke else "real", "n": int(X.shape[0]),
           "oracle_k": oracle, "seeds": SEEDS, "threads": t, "env": env_summary()}
    t0 = time.perf_counter()
    if "e1" in run: out["E1_divisive"] = exp1(X, labels, oracle)
    if "e2" in run: out["E2_ablation"] = exp2(X, labels, oracle)
    if "e3" in run: out["E3_pca_sweep"] = exp3(X, labels, oracle)
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)

    kind = "smoke" if args.smoke else ("subsample" if args.n and args.n < n else "real")
    dst = EXPERIMENTS_OUT / f"cub_experiments_{kind}.json"           # step 4 reads only the full-run files
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"  wrote {dst}  ({out['elapsed_s']}s)", flush=True)

    if args.smoke:
        sp = out.get("E1_divisive", {}).get("species", {}).get("flat_kmeans", {}) \
             or out.get("E2_ablation", {}).get("kmeans", {}).get("species", {})
        if "ari" in sp:
            print(f"  [smoke] species ARI={sp['ari']['mean']:.3f} (must be ~0), "
                  f"NMI={sp['nmi']['mean']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
