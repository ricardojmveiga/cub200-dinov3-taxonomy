#!/usr/bin/env python3
"""Draw the README's two overview figures from the repository's own data.

  docs/tree.png     the paper's Ward clustering drawn as a tree: Ward builds one tree over the 11,788
                    images; cut into 13, 37, 121 and 200 groups, it is the Ward row of Table 1 (the
                    script checks it reproduces those ARIs). Drawn radially down to its 200 groups:
                    rim tile = the order most images of that group belong to; branches coloured by
                    order where all groups below share one, grey where orders mix; dashed rings = the
                    cuts into 13, 37 and 121 groups; shaded sectors = the 13 groups. No photos.
                    Needs the embeddings (python fetch_embeddings.py).
  docs/methods.png  Table 1 as dots: one per clusterer and rank (mean ARI), from the bundled results.

    python docs/make_readme_figures.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Wedge
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score as ari

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reprocub import common  # noqa: E402

OUT = ROOT / "docs"
RANKS = ("order", "family", "genus", "species")
E2 = json.loads((common.EXPERIMENTS_OUT / "cub_experiments_e2.json").read_text(encoding="utf-8"))["E2_ablation"]
NAMES = json.loads((common.DATA / "cub200_order_names.json").read_text(encoding="utf-8"))["orders"]
SP_ORDER = np.load(common.DATA / "cub_species_to_order.npy")
BY_SIZE = [o for o, _ in Counter(SP_ORDER).most_common()]                      # as in Figure 2
PALETTE = ["#0b2e63", "#e6194b", "#f58231", "#3cb44b", "#4363d8", "#911eb4", "#f032e6",
           "#469990", "#9a6324", "#800000", "#808000", "#ffd000", "#ff69b4"]
COL = {o: PALETTE[k] for k, o in enumerate(BY_SIZE)}
INK, MUT, GREY = "#1a1a1a", "#5f5f5f", "#a8a8a8"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})


def methods_figure() -> None:
    name = {"kmeans": "k-means", "kmeans_whiten": "standardised k-means", "bisecting": "bisecting k-means",
            "ward": "Ward", "gmm_diag": "GMM", "spectral": "spectral"}
    _, oracle, _ = common.load_labels()
    blue, red = "#0072B2", "#D55E00"
    fig, ax = plt.subplots(figsize=(5.6, 3.6))          # small canvas: shown at half the README width
    rows = [("species", "Species"), ("genus", "Genus"), ("family", "Family"), ("order", "Order")]
    for y, (rank, label) in enumerate(reversed(rows)):
        vals = {m: E2[m][rank]["ari"]["mean"] for m in name}
        lo, hi = min(vals.values()), max(vals.values()); best = max(vals, key=vals.get)
        c = red if rank == "order" else blue
        ax.plot([lo, hi], [y, y], color=c, lw=11, alpha=0.22, solid_capstyle="round", zorder=1)
        ax.scatter(list(vals.values()), [y] * len(vals), s=80, color=c, edgecolor="white", linewidth=1.4, zorder=3)
        ax.text(hi + 0.025, y + 0.06, f"{hi:.3f}", va="center", ha="left", fontsize=13, fontweight="bold", color=c)
        ax.text(hi + 0.025, y - 0.32, f"best: {name[best]}", va="center", ha="left", fontsize=9.5, color=MUT)
        ax.text(-0.07, y + 0.06, label, va="center", ha="right", fontsize=13, fontweight="bold", color=INK)
        ax.text(-0.07, y - 0.32, f"{oracle[rank]} groups", va="center", ha="right", fontsize=9.5, color=MUT)
    ax.axvline(0, color="#8d939c", lw=1.2, ls=(0, (5, 4)), zorder=0)
    ax.set_xlim(-0.33, 1.02); ax.set_ylim(-0.6, 3.55); ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8]); ax.set_xticklabels(["0\nrandom", "0.2", "0.4", "0.6", "0.8"])
    ax.set_xlabel("agreement with the taxonomy (ARI), one dot per clusterer", fontsize=10.5, color=INK, labelpad=5)
    ax.tick_params(axis="x", labelsize=10, length=4, color="#8d939c")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#8d939c"); ax.spines["bottom"].set_bounds(0, 0.8)
    fig.savefig(OUT / "methods.png", dpi=190, bbox_inches="tight", pad_inches=0.15, facecolor="white",
                metadata={"Software": None}); plt.close(fig)
    print("wrote", OUT / "methods.png")


def tree_figure() -> None:
    labels, oracle, n = common.load_labels()
    if not common.EMBEDDINGS_NPY.exists():
        print("  tree.png needs the embeddings: run python fetch_embeddings.py first"); return
    X = np.load(common.EMBEDDINGS_NPY).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Z = linkage(X, "ward"); del X                       # what AgglomerativeClustering(linkage="ward") builds
    cut = {r: fcluster(Z, oracle[r], "maxclust") for r in RANKS}
    ARI = {r: float(ari(labels[r], cut[r])) for r in RANKS}
    for r in RANKS:
        assert abs(ARI[r] - E2["ward"][r]["ari"]["mean"]) < 1.1e-3, f"Ward at {r}: {ARI[r]:.4f} is not Table 1's value"
    print("Ward cuts reproduce Table 1:", {r: round(ARI[r], 3) for r in RANKS})

    K = oracle["species"]; first = n - K; LEAF = n + first
    kids = {n + j: (int(Z[j, 0]), int(Z[j, 1])) for j in range(first, n - 1)}

    def any_image(v):
        while v >= n:
            v = int(Z[v - n, 0])
        return v

    tips, stack = [], [2 * n - 2]
    while stack:
        v = stack.pop()
        if v < LEAF:
            tips.append(v)
        else:
            a, b = kids[v]; stack += [b, a]
    pos = {v: i for i, v in enumerate(tips)}
    members = {v: np.flatnonzero(cut["species"] == cut["species"][any_image(v)]) for v in tips}
    tip_order = {v: Counter(labels["order"][members[v]]).most_common(1)[0][0] for v in tips}
    span, below, k_after = {}, {}, {}
    for v in tips:
        span[v] = (pos[v], pos[v]); below[v] = {tip_order[v]}
    for j in range(first, n - 1):
        v = n + j; a, b = kids[v]
        span[v] = (min(span[a][0], span[b][0]), max(span[a][1], span[b][1]))
        below[v] = below[a] | below[b]; k_after[v] = n - 1 - j
    g13 = cut["order"]; top = {}
    for v in tips:
        top.setdefault(g13[any_image(v)], []).append(v)
    sectors = sorted((min(pos[v] for v in vs), len(vs)) for vs in top.values())

    R, GAP = 9.0, np.deg2rad(34)
    ang = lambda p: np.pi / 2 - (GAP / 2 + (p + 0.5) / K * (2 * np.pi - GAP))
    rad = lambda k: R * np.log(k + 0.7) / np.log(K + 0.7)
    node_r = lambda v: R if v < LEAF else rad(k_after[v])
    node_a = lambda v: ang((span[v][0] + span[v][1]) / 2)
    node_c = lambda v: COL[next(iter(below[v]))] if len(below[v]) == 1 else GREY
    xy = lambda r, a: (r * np.cos(a), r * np.sin(a))

    fig = plt.figure(figsize=(13.5, 10.6))
    ax = fig.add_axes([0.0, 0.0, 0.78, 1.0]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-10.8, 10.6); ax.set_ylim(-10.3, 10.6)
    for k, (s0, m) in enumerate(sectors):                  # the model's 13 groups, as alternating sectors
        a1, a0 = ang(s0 - 0.5), ang(s0 + m - 0.5)
        ax.add_patch(Wedge((0, 0), R * 1.115, np.rad2deg(a0), np.rad2deg(a1), width=R * 1.115 - rad(12.5),
                           fc="#eef1f5" if k % 2 == 0 else "#ffffff", ec="none", zorder=0))
        for a in (a0, a1):
            ax.plot(*zip(xy(rad(12.5), a), xy(R * 1.115, a)), color="#c4c9d1", lw=0.8, zorder=1)
    for r in ("order", "family", "genus"):                 # the cuts into 13, 37 and 121 groups
        t = np.linspace(ang(-0.5), ang(K - 0.5), 400)
        ax.plot(rad(oracle[r] - 0.5) * np.cos(t), rad(oracle[r] - 0.5) * np.sin(t), color="#8d939c", lw=1.0, ls=(0, (6, 5)), zorder=1)
    for v in kids:                                         # branches
        a, b = kids[v]; r0 = node_r(v)
        t = np.linspace(node_a(a), node_a(b), max(8, int(abs(node_a(a) - node_a(b)) * 60)))
        ax.plot(r0 * np.cos(t), r0 * np.sin(t), color=node_c(v), lw=1.3, solid_capstyle="round", zorder=3)
        for c in (a, b):
            ax.plot(*zip(xy(r0, node_a(c)), xy(node_r(c), node_a(c))), color=node_c(c), lw=1.3, solid_capstyle="round", zorder=3)
    for v in tips:                                         # rim: one tile per group, in its order's colour
        a1, a0 = ang(pos[v] - 0.5), ang(pos[v] + 0.5)
        ax.add_patch(Wedge((0, 0), R * 1.10, np.rad2deg(a0), np.rad2deg(a1), width=R * 0.075,
                           fc=COL[tip_order[v]], ec="white", lw=0.4, zorder=4))
    for r, word in (("order", "orders"), ("family", "families"), ("genus", "genera"), ("species", "species")):
        rr = R * 1.06 if r == "species" else rad(oracle[r] - 0.5)
        ax.text(0, rr, f"{oracle[r]} groups vs {word}: {ARI[r]:.3f}", ha="center", va="center", fontsize=11.5,
                fontweight="bold", color=COL[BY_SIZE[1]] if r == "order" else INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.94))

    lg = fig.add_axes([0.77, 0.10, 0.23, 0.80]); lg.axis("off"); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
    lg.text(0.0, 0.97, "Colour = the\nbiologists' order", fontsize=14, fontweight="bold", color=INK, va="top")
    counts = Counter(labels["order"])
    for i, o in enumerate(BY_SIZE):
        y = 0.83 - i * 0.052
        lg.add_patch(Rectangle((0.0, y - 0.015), 0.07, 0.03, fc=COL[o], ec="#333", lw=0.4))
        lg.text(0.10, y, f"{NAMES[o]}  ({counts[o]:,})", va="center", fontsize=11, color=INK)
    lg.text(0.0, 0.83 - 13 * 0.052, "grey branch = orders mixed\n(images per order in brackets)", fontsize=10, color=MUT, va="top")
    fig.savefig(OUT / "tree.png", dpi=110, facecolor="white", metadata={"Software": None}); plt.close(fig)
    print("wrote", OUT / "tree.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    methods_figure()
    tree_figure()
