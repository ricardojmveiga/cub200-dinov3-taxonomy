"""Clustering primitives + the E1/E2/E3 studies (importable, CPU-only, deterministic).

Kept separate from the CLI (02_run_experiments.py) so the tests can call the clusterers
directly on a tiny random matrix -- the built-in sanity check that ARI ~ 0 on noise while
NMI is inflated (Vinh 2010) -- without running the full multi-seed sweep.
"""
from __future__ import annotations

import numpy as np

SEEDS = [42, 123, 456, 789, 2024]
PCA_DIMS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
RANKS = ("order", "family", "genus", "species")


def evalp(yt, yp) -> dict:
    from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
    return {"nmi": float(nmi(yt, yp)), "ari": float(ari(yt, yp))}


def agg(dicts) -> dict:
    out = {}
    for m in ("nmi", "ari"):
        a = np.array([d[m] for d in dicts], float)
        out[m] = {"mean": float(a.mean()), "std": float(a.std()),
                  "runs": [round(float(v), 4) for v in a]}
    return out


def km(X, k, seed):
    from sklearn.cluster import MiniBatchKMeans
    return MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10, batch_size=1024).fit_predict(X)


def km_whiten(X, k, seed):
    """Per-dimension standardisation (StandardScaler), then k-means: the paper's 'standardised
    k-means' (not PCA whitening)."""
    from sklearn.preprocessing import StandardScaler
    return km(StandardScaler().fit_transform(X), k, seed)


def ward(X, k, seed):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def gmm(X, k, seed):
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(n_components=k, covariance_type="diag", random_state=seed,
                           max_iter=100).fit_predict(X)


def spectral(X, k, seed):
    from sklearn.cluster import SpectralClustering
    return SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=15,
                              random_state=seed, assign_labels="kmeans").fit_predict(X)


def bisecting(X, k, seed):
    from sklearn.cluster import BisectingKMeans
    return BisectingKMeans(n_clusters=k, random_state=seed,
                           bisecting_strategy="largest_cluster").fit_predict(X)


METHODS = {"kmeans": (km, 5), "kmeans_whiten": (km_whiten, 5), "ward": (ward, 1),
           "gmm_diag": (gmm, 3), "spectral": (spectral, 2), "bisecting": (bisecting, 5)}


def _runs(fn, X, labels, r, k, seeds):
    return agg([evalp(labels[r], fn(X, k, s)) for s in seeds])


def exp1(X, labels, oracle) -> dict:
    res = {}
    for r in RANKS:
        k = oracle[r]
        res[r] = {"oracle_K": k, "flat_kmeans": _runs(km, X, labels, r, k, SEEDS),
                  "divisive_bisecting": _runs(bisecting, X, labels, r, k, SEEDS)}
    return res


def exp2(X, labels, oracle, verbose=True) -> dict:
    res = {}
    for name, (fn, ns) in METHODS.items():
        res[name] = {}
        for r in RANKS:
            try:
                res[name][r] = _runs(fn, X, labels, r, oracle[r], SEEDS[:ns])
            except Exception as e:
                # Keep the sweep alive if one clusterer dies, but make the hole LOUD: a
                # silently dropped method would vanish from Table 1 (rendered as '--') while
                # the run still exits 0, so a partial re-run could masquerade as a full one.
                print(f"  !! WARNING: E2 {name}/{r} FAILED -- {str(e)[:140]}", flush=True)
                res[name][r] = {"error": str(e)[:140]}
        if verbose:
            print("  E2 %-14s %s" % (name, "  ".join(
                f"{r}:{res[name][r].get('ari', {}).get('mean', float('nan')):.3f}" for r in RANKS)), flush=True)
    return res


def exp3(X, labels, oracle) -> dict:
    from sklearn.decomposition import PCA
    maxd = min(X.shape)
    dims = [d for d in PCA_DIMS if d <= maxd]
    pca = PCA(n_components=min(max(dims), maxd), random_state=0).fit(X)
    Xp = pca.transform(X)
    res = {"dims": dims, "per_rank": {r: {} for r in RANKS},
           "explained_var_cumsum": [float(v) for v in
                                    np.cumsum(pca.explained_variance_ratio_)[[d - 1 for d in dims]]]}
    for d in dims:
        for r in RANKS:
            res["per_rank"][r][str(d)] = _runs(km, Xp[:, :d], labels, r, oracle[r], SEEDS)
    return res


def control(labels, oracle, seeds=SEEDS):
    """Random-feature control at the REAL oracle K. Draws L2-normalised Gaussian features of the
    same (n, 4096) shape as the frozen embeddings and clusters them by flat MiniBatch k-means at
    each rank's true K. ARI must be ~0 (chance-corrected) at EVERY rank; at this n the NMI stays
    near zero too (0.004-0.037). This backs the paper's "a random-feature control gives ARI~0" at
    the reported K={13,37,121,200}, not at the smaller K of the fast --smoke gate."""
    n = len(labels[RANKS[0]])
    per = {r: [] for r in RANKS}
    for s in seeds:                                          # fresh random features per seed
        rng = np.random.default_rng(s)
        X = rng.standard_normal((n, 4096)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        for r in RANKS:
            per[r].append(evalp(labels[r], km(X, oracle[r], s)))
    return {r: agg(per[r]) for r in RANKS}


def load_embeddings(path, n, smoke):
    if smoke:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 4096)).astype(np.float32)
        return X / np.linalg.norm(X, axis=1, keepdims=True)
    X = np.load(path).astype(np.float32)
    assert X.shape[0] == n, f"embeddings {X.shape} vs {n} labels"
    return X
