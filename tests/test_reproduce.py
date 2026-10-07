"""The manuscript's Figure 2 numbers, Table 1 cells and figure files reproduce from data/."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from reprocub.common import DATA, EXPERIMENTS_OUT, FIGURES


def _within(S, lab):
    N = len(lab)
    m = (lab[:, None] == lab[None, :]) & ~np.eye(N, dtype=bool)
    return float(S[m].mean())


def test_within_rank_cosines_match_manuscript():
    C = np.load(DATA / "cub_species_centroids_4096.npy").astype(np.float64)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    S = C @ C.T
    o = np.load(DATA / "cub_species_to_order.npy")
    f = np.load(DATA / "cub_species_to_family.npy")
    g = np.load(DATA / "cub_species_to_genus.npy")
    assert abs(_within(S, g) - 0.663) < 0.01                 # within-Genus  (manuscript 0.66)
    assert abs(_within(S, f) - 0.389) < 0.01                 # within-Family (0.39)
    assert abs(_within(S, o) - 0.092) < 0.01                 # within-Order  (0.09 ~ chance)


def test_table1_cells_match_manuscript():
    e2 = json.loads((EXPERIMENTS_OUT / "cub_experiments_e2.json").read_text(encoding="utf-8"))["E2_ablation"]
    ari = lambda m, r: e2[m][r]["ari"]["mean"]
    assert abs(ari("spectral", "species") - 0.739) < 0.001   # bold best-species
    assert abs(ari("ward", "genus") - 0.587) < 0.001         # bold best-genus
    assert abs(ari("gmm_diag", "family") - 0.460) < 0.001    # bold best-family
    assert abs(ari("kmeans", "order") - 0.092) < 0.001       # near-chance Order
    assert ari("spectral", "order") < 0                      # spectral Order is negative


def test_figures_regenerate_from_bundled_data():
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "04_make_figures.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert (FIGURES / "fig_simmatrix.png").exists()
    assert (FIGURES / "fig_pca_sweep.pdf").exists()
