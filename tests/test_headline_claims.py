"""Every headline number the manuscript reports must reproduce from the bundled data.

Before this file the suite guarded only 5 of the 24 Table 1 cells and *none* of the
paper's lead claims (kNN@1@Order, the PCA sweep, the 67% order imbalance, the chance
baseline). These tests pin them to their source JSON / npy so a silent drift can't ship.

All are CI-safe (run from the bundled data/) except ``test_knn_at1_order_*``, which needs
the 193 MB embeddings and self-skips when they are absent (e.g. on CI).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from reprocub.common import DATA, EMBEDDINGS_NPY, EXPERIMENTS_OUT, load_labels

E2 = json.loads((EXPERIMENTS_OUT / "cub_experiments_e2.json").read_text(encoding="utf-8"))["E2_ablation"]
E13 = json.loads((EXPERIMENTS_OUT / "cub_experiments_e1e3.json").read_text(encoding="utf-8"))
RANKS = ("order", "family", "genus", "species")

# exact ARI means the manuscript's Table 1 prints (method -> [order, family, genus, species])
TABLE1 = {
    "kmeans":        [0.092, 0.435, 0.553, 0.672],
    "kmeans_whiten": [0.084, 0.440, 0.571, 0.667],
    "bisecting":     [0.077, 0.407, 0.515, 0.710],
    "ward":          [0.012, 0.376, 0.587, 0.703],
    "gmm_diag":      [0.073, 0.460, 0.543, 0.649],
    "spectral":     [-0.025, 0.333, 0.279, 0.739],
}


@pytest.mark.parametrize("method", list(TABLE1))
def test_all_table1_cells_match_manuscript(method):
    for r, want in zip(RANKS, TABLE1[method]):
        got = E2[method][r]["ari"]["mean"]
        assert abs(got - want) < 1.1e-3, f"{method}/{r}: json {got:.4f} vs tex {want}"


def test_e2_ablation_complete_no_silent_failures():
    """Guards the loud-warning fix in exp2: every method x rank must have real ARI stats,
    never an {'error': ...} cell (a silently dropped clusterer would ship as '--')."""
    for method in TABLE1:
        for r in RANKS:
            cell = E2[method][r]
            assert "error" not in cell, f"{method}/{r} failed at generation: {cell.get('error')}"
            assert "ari" in cell and "mean" in cell["ari"], f"{method}/{r} malformed"


def test_pca_sweep_species_peak_and_order_ceiling():
    pr = E13["E3_pca_sweep"]["per_rank"]
    sp = {int(d): v["ari"]["mean"] for d, v in pr["species"].items()}
    gn = {int(d): v["ari"]["mean"] for d, v in pr["genus"].items()}
    od = {int(d): v["ari"]["mean"] for d, v in pr["order"].items()}
    assert max(sp, key=sp.get) == 256                   # SweepSpeciesPeakDim
    assert abs(sp[256] - 0.681) < 1.1e-3                 # SweepSpeciesPeak
    assert abs(sp[4096] - 0.672) < 1.1e-3               # SweepSpeciesFull
    assert max(gn, key=gn.get) == 128                    # SweepGenusPeakDim
    assert max(od.values()) <= 0.097 + 1.1e-3           # "never above 0.097"
    assert round(max(od.values()), 3) == 0.097          # SweepOrderPeak
    assert abs(od[4096] - 0.092) < 1.1e-3               # FlatOrderARI
    # Results: "Order hovers near zero, falling below chance across dimensions 16--512"
    # pin the whole asserted range, not just its minimum, or the claim ships unguarded
    assert all(od[d] < 0 for d in (16, 32, 64, 128, 256, 512)), \
        f"order ARI not negative across 16-512: {[(d, round(od[d], 4)) for d in sorted(od)]}"
    assert all(od[d] > 0 for d in (1024, 2048, 4096))    # and back above chance beyond 512
    assert od[128] == min(od.values())                  # 128 is where it bottoms out


def test_pca_sweep_family_dip_near_128():
    """Results: 'The Family curve is conspicuously non-monotonic near dimension 128.'

    The whole Family sweep curve -- one of the four lines in Figure 1 -- had no coverage,
    so the only claim the manuscript makes about it could drift unnoticed. Pin the local dip.
    """
    fam = {int(d): v["ari"]["mean"] for d, v in E13["E3_pca_sweep"]["per_rank"]["family"].items()}
    assert fam[128] < fam[64] and fam[128] < fam[256], (
        f"family ARI not dipping at 128: 64={fam[64]:.3f} 128={fam[128]:.3f} 256={fam[256]:.3f}")


def test_pca_sweep_dimensionalities_stated_in_the_conclusion():
    """Conclusion: 'species and genus need only 256 of the 4096
    principal components and family about 1024, whereas coarse Order recovery stays near chance
    across six clusterers and every dimensionality, the full 4096 included.'

    256 and 4096 are macros, but '1024' is typed by hand. Pin what the sentence claims: by 256
    dimensions species and genus already match their full-dimensional score, family does not get
    there before 1024, and Order never leaves the neighbourhood of chance, at 4096 or anywhere else.
    """
    pr = E13["E3_pca_sweep"]["per_rank"]
    ari = {r: {int(d): v["ari"]["mean"] for d, v in pr[r].items()} for r in RANKS}
    for r in ("species", "genus"):
        assert ari[r][256] >= 0.99 * ari[r][4096], f"{r}: 256-D {ari[r][256]:.3f} vs full {ari[r][4096]:.3f}"
    fam = ari["family"]
    assert fam[1024] >= 0.97 * fam[4096], f"family at 1024 ({fam[1024]:.3f}) is not near full ({fam[4096]:.3f})"
    early = {d: round(v, 3) for d, v in fam.items() if d < 1024 and v >= 0.90 * fam[4096]}
    assert not early, f"family already near its full-dimensional score below 1024: {early}"
    assert max(ari["order"].values()) < 0.1          # near chance at every dimensionality, 4096 included


def test_best_order_ari_is_the_column_max():
    """\\BestOrderARI = 0.092 is stated as the best Order score across all six clusterers.

    Every Order cell is pinned individually, so 'best' was only an emergent property of six
    separate literals; assert the intent directly.
    """
    order_means = {m: E2[m]["order"]["ari"]["mean"] for m in TABLE1}
    assert max(order_means, key=order_means.get) == "kmeans"
    assert round(max(order_means.values()), 3) == 0.092


# The paper's 24 result macros, copied verbatim from its LaTeX preamble (the paper's source is
# not part of this repository).
PAPER_MACROS = DATA / "paper_macros.tex"


def test_manuscript_macros_match_generated_values():
    """Close the last drift gap: until now no test ever read the manuscript itself.

    Every reported number is a ``\\newcommand`` in the paper's preamble, pasted from the
    generator's output; data/paper_macros.tex holds those lines verbatim. Each headline test re-types its own expected literal, so a hand
    edit that diverged from BOTH the JSON *and* every test literal would ship silently.
    Parse the .tex and diff it against the values the generator derives from the bundled
    result JSONs. The generator emits a superset of what the paper uses, so cross-check
    the intersection (and require it to be non-trivial, so this cannot pass vacuously).
    """
    import re

    from reprocub.fig_sweep_table import load, macros

    pat = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}")
    tex = dict(pat.findall(PAPER_MACROS.read_text(encoding="utf-8")))
    assert len(tex) == 24, f"expected the paper's 24 macros, found {len(tex)}"
    generated = dict(pat.findall("\n".join(macros(load()))))
    assert generated, "generator produced no macros"

    shared = sorted(set(tex) & set(generated))
    assert len(shared) >= 12, f"only {len(shared)} macros cross-checked ({shared}) -- too few to trust"
    drifted = {k: {"tex": tex[k], "generated": generated[k]} for k in shared if tex[k] != generated[k]}
    assert not drifted, f"manuscript macros disagree with the result JSONs: {drifted}"

    # the label-derived constants the generator does not emit
    _, oracle, n = load_labels()
    assert tex["Nimg"].replace(",", "").replace("{,}", "") == str(n)          # 11,788
    assert tex["Nspecies"] == str(oracle["species"]) == tex["KSpecies"]       # 200
    for rank, macro in (("order", "KOrder"), ("family", "KFamily"), ("genus", "KGenus")):
        assert tex[macro] == str(oracle[rank]), f"{macro}={tex[macro]} vs oracle {oracle[rank]}"


def test_methods_hyperparameters_match_prose():
    """Methods names two clusterer hyperparameters in prose; both were only guarded
    indirectly (via the ARI cells), so a numerically-compensating change could ship silently."""
    import inspect

    from reprocub import experiments as ex

    assert "n_init=10" in inspect.getsource(ex.km)            # "$n_init{=}10$"
    assert "n_neighbors=15" in inspect.getsource(ex.spectral)  # "15-nearest-neighbour graph"


def test_flat_kmeans_nmi_ari_pairs_match_manuscript():
    """The NMI-inflation argument: high NMI but chance-level ARI at both ends."""
    e1 = E13["E1_divisive"]
    sp, od = e1["species"]["flat_kmeans"], e1["order"]["flat_kmeans"]
    assert abs(sp["nmi"]["mean"] - 0.909) < 1.1e-3 and abs(sp["ari"]["mean"] - 0.672) < 1.1e-3
    assert abs(od["nmi"]["mean"] - 0.410) < 1.1e-3 and abs(od["ari"]["mean"] - 0.092) < 1.1e-3


def test_spectral_order_std_and_runs():
    sp = E2["spectral"]["order"]["ari"]
    assert round(sp["std"], 2) == 0.06                  # OrderStdSpectral
    assert sp["mean"] < 0                                # spectral Order is negative


def test_spectral_species_std_is_negligible():
    """The headline number (\\BestSpeciesARI = 0.739) comes from spectral, the method with the
    THINNEST seed budget (2) -- and the same sentence calls that estimate noisy at Order. It is
    nonetheless effectively deterministic at species, which is why the manuscript can lead with
    it. Pinned so that claim cannot rot."""
    sp = E2["spectral"]["species"]["ari"]
    assert sp["std"] < 1e-3, f"spectral/species std {sp['std']} -- headline no longer stable"
    assert len(set(round(r, 4) for r in sp["runs"])) == 1   # both seeds land on 0.7389


def test_explained_variance_monotonic_and_full():
    sw = E13["E3_pca_sweep"]
    ev, dims = sw["explained_var_cumsum"], sw["dims"]
    assert len(ev) == 12
    assert all(a <= b + 1e-9 for a, b in zip(ev, ev[1:]))
    assert 0 < ev[0] < 1 and abs(ev[-1] - 1.0) < 1e-3
    assert abs(ev[dims.index(256)] - 0.914) < 2e-3      # 256 dims ~= 91% variance


def test_majority_order_fraction_is_67pct():
    labels, _, n = load_labels()
    counts = np.bincount(labels["order"])
    assert int(counts.max()) == 7900 and n == 11788
    assert round(100 * counts.max() / n) == 67          # MajorityOrderPct


def test_centroid_offdiagonal_chance_baseline():
    """Discussion: within-order cosine 0.09 is 'barely above the chance 0.05'."""
    C = np.load(DATA / "cub_species_centroids_4096.npy").astype(np.float64)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    S = C @ C.T
    off = S[np.triu_indices(len(C), 1)].mean()
    assert abs(off - 0.05) < 0.02                        # global chance ~ 0.05


def test_within_rank_centroid_cosines():
    """Figure 2 / Discussion: the fine-coarse dissociation *as measured in 4096-D*.

    The manuscript quotes 'within-genus cosine 0.66' and 'within-order 0.09, barely above
    the chance 0.05'; reprocub/fig_simmatrix.py prints 0.092 / 0.389 / 0.663 / 0.052. Only the
    global off-diagonal was pinned before, so the two figures actually cited in prose could
    drift silently. Pin all four to the values the figure caption and Discussion rest on.
    """
    C = np.load(DATA / "cub_species_centroids_4096.npy").astype(np.float64)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    S = C @ C.T
    eye = np.eye(len(C), dtype=bool)

    def within(lab):
        m = (lab[:, None] == lab[None, :]) & ~eye
        return S[m].mean()

    expected = {"order": 0.092, "family": 0.389, "genus": 0.663}
    for rank, want in expected.items():
        lab = np.load(DATA / f"cub_species_to_{rank}.npy")
        got = within(lab)
        assert abs(got - want) < 1.1e-3, f"within-{rank} cosine {got:.4f} != {want}"

    off = S[np.triu_indices(len(C), 1)].mean()
    assert abs(off - 0.052) < 1.1e-3                     # global off-diagonal (chance)
    # the dissociation itself: genus >> family >> order ~= chance
    assert within(np.load(DATA / "cub_species_to_genus.npy")) > \
           within(np.load(DATA / "cub_species_to_family.npy")) > \
           within(np.load(DATA / "cub_species_to_order.npy"))


def test_random_feature_control_ari_near_zero_at_real_K():
    """Random-feature control at the REAL oracle K (n=11788, K=13/37/121/200) -> ARI ~ 0 at every
    rank, backing the Methods 'a random-feature control gives ARI~0' claim. NB the control's NMI
    is ALSO near zero at this n (0.004-0.037), i.e. NOT 'inflated' -- inflation needs few samples
    per cluster (the small-n --smoke gate), so the paper does not claim inflated control NMI."""
    import json
    c = json.loads((EXPERIMENTS_OUT / "cub_experiments_control.json").read_text(encoding="utf-8"))
    assert c["mode"] == "control" and c["n"] == 11788
    for r in RANKS:
        ari = c["per_rank"][r]["ari"]
        assert abs(ari["mean"]) < 0.01, f"{r} control ARI mean {ari['mean']}"
        assert max(abs(v) for v in ari["runs"]) < 0.01, f"{r} control ARI runs {ari['runs']}"
        assert c["per_rank"][r]["nmi"]["mean"] < 0.10   # NMI near zero at real K (not inflated)


@pytest.mark.skipif(not EMBEDDINGS_NPY.exists(),
                    reason="193 MB embeddings absent (run fetch_embeddings.py); skipped in CI")
def test_knn_at1_order_is_99_9_percent():
    # exercise the actual step-3 implementation (module name starts with a digit)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    knn1_order = __import__("03_compute_centroids", fromlist=["knn1_order"]).knn1_order
    X = np.load(EMBEDDINGS_NPY).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    labels, _, n = load_labels()
    correct, N = knn1_order(X, labels["order"], "cpu")
    assert N == n == 11788 and correct == 11778
    assert round(100 * correct / N, 1) == 99.9          # BirdOrderkNN
