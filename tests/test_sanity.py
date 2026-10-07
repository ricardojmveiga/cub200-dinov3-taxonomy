"""Built-in falsification gate: random embeddings must give ARI ~ 0 (NMI inflated)."""
import numpy as np
from reprocub.experiments import evalp, km


def _rand_unit(n, d, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_random_data_ari_near_zero():
    X = _rand_unit(400, 64)
    y = np.random.default_rng(1).integers(0, 13, 400)        # random 'true' labels
    r = evalp(y, km(X, 13, 42))
    assert abs(r["ari"]) < 0.06, r                           # chance-corrected -> ~0


def test_nmi_inflated_at_fine_ranks():
    X = _rand_unit(400, 64)
    y = np.random.default_rng(2).integers(0, 120, 400)       # many classes, few samples
    r = evalp(y, km(X, 120, 42))
    assert r["nmi"] > r["ari"] + 0.15                        # NMI biased high, ARI ~ 0
    assert abs(r["ari"]) < 0.06
