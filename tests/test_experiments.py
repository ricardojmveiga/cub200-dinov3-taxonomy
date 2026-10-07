"""Every clusterer and the E1/E3 drivers must run (on tiny smoke data).

test_sanity only ever exercised MiniBatch k-means; the other five clusterers, the smoke
loader, the aggregation helper and the exp1/exp3 wiring had no coverage, so a broken
scikit-learn call or a config regression would pass CI unnoticed.
"""
import warnings

import numpy as np
import pytest

from reprocub.experiments import (METHODS, PCA_DIMS, SEEDS, agg, evalp, exp1, exp3,
                                  load_embeddings)

# per-method seed budgets the manuscript discloses (kmeans 5, whiten 5, ward 1, gmm 3,
# spectral 2, bisecting 5) -- second element of each METHODS entry.
BUDGETS = {"kmeans": 5, "kmeans_whiten": 5, "ward": 1, "gmm_diag": 3, "spectral": 2, "bisecting": 5}


def _smoke(n=300, d=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.mark.parametrize("name", list(METHODS))
def test_every_clusterer_runs_and_is_chance_on_noise(name):
    warnings.filterwarnings("ignore")
    fn, _ = METHODS[name]
    X = _smoke()
    y = np.random.default_rng(1).integers(0, 10, len(X))
    labels = fn(X, 10, 42)
    assert labels.shape == (len(X),)
    assert len(set(labels.tolist())) == 10               # produced K clusters
    assert abs(evalp(y, labels)["ari"]) < 0.08           # chance on noise


def test_seeds_and_method_budgets_match_manuscript():
    assert SEEDS == [42, 123, 456, 789, 2024]
    assert {m: ns for m, (_, ns) in METHODS.items()} == BUDGETS
    assert PCA_DIMS == [2 ** i for i in range(1, 13)]     # 2 .. 4096


def test_agg_reports_mean_std_runs():
    a = agg([{"nmi": 0.5, "ari": 0.1}, {"nmi": 0.7, "ari": 0.3}])
    assert abs(a["ari"]["mean"] - 0.2) < 1e-9
    assert len(a["ari"]["runs"]) == 2 and a["nmi"]["std"] > 0


def test_load_embeddings_smoke_is_unit_norm():
    X = load_embeddings(None, 300, smoke=True)
    assert X.shape == (300, 4096)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)


def test_load_embeddings_rejects_wrong_row_count(tmp_path):
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((17, 4096), dtype=np.float32))
    with pytest.raises(AssertionError):
        load_embeddings(bad, 11788, smoke=False)         # 17 rows != 11788 labels


def test_exp1_and_exp3_wiring_runs_on_smoke():
    warnings.filterwarnings("ignore")
    X = load_embeddings(None, 200, smoke=True)
    rng = np.random.default_rng(3)
    oracle = {"order": 4, "family": 6, "genus": 8, "species": 10}
    labels = {r: rng.integers(0, k, 200) for r, k in oracle.items()}
    r1 = exp1(X, labels, oracle)
    assert set(r1) == set(oracle) and "flat_kmeans" in r1["order"]
    r3 = exp3(X[:, :64], labels, oracle)                 # 64-D so PCA is quick
    assert r3["dims"] == [d for d in PCA_DIMS if d <= 64]
    assert set(r3["per_rank"]) == set(oracle)
    assert len(r3["explained_var_cumsum"]) == len(r3["dims"])
