#!/usr/bin/env python3
"""Taxonomy-ordered cosine-similarity matrix of the 200 CUB-200 per-species centroids in the
REAL 4096-D frozen-DINOv3 (CLS, L2-normalised) space. Rows/cols are sorted Order>Family>Genus>
species, so the ground-truth taxonomy forms nested blocks down the diagonal. The BLOCK BOUNDARIES
are imposed by the sort; the BRIGHTNESS is the measured DINOv3 cosine. This renders the fine-coarse
dissociation directly and honestly:

  - bright fine (genus/family) blocks tile the diagonal  -> fine ranks are recovered;
  - the 67% Passeriformes region forms NO bright order-level super-block (within-order cosine
    ~= global off-diagonal) -> the coarse rank is not globally separable.

It is a plain cosine (Gram) matrix of the species centroids: no clustering, no cut, no free
parameter. The sort biases TOWARD showing an order super-block, so its ABSENCE is conservative
evidence.
Writes figures/fig_simmatrix.{pdf,png}."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
from collections import Counter

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))
from common import DATA as _DATA, FIGURES as _FIG, EXPERIMENTS_OUT as _OUT
HERE = str(_DATA.parent)
DATA = str(_DATA)
C = np.load(os.path.join(DATA, "cub_species_centroids_4096.npy")).astype(np.float64)
C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
order = np.load(os.path.join(DATA, "cub_species_to_order.npy"))
family = np.load(os.path.join(DATA, "cub_species_to_family.npy"))
genus = np.load(os.path.join(DATA, "cub_species_to_genus.npy"))
names = json.load(open(os.path.join(DATA, "cub200_order_names.json"), encoding="utf-8"))["orders"]
N = len(C)

S = C @ C.T                                                    # 200x200 cosine similarity (plain Gram)
perm = np.lexsort((np.arange(N), genus, family, order))        # sort Order>Family>Genus>species
Sp = S[np.ix_(perm, perm)]
o_s, f_s = order[perm], family[perm]

# within-rank mean cosines (reported in the caption; NOT eyeballed)
iu = np.triu_indices(N, 1)
gl = S[iu].mean()
def within(lab):
    m = (lab[:, None] == lab[None, :]) & ~np.eye(N, dtype=bool)
    return S[m].mean()
print(f"mean cosine  within-Order {within(order):.3f}  within-Family {within(family):.3f}  "
      f"within-Genus {within(genus):.3f}  global-off {gl:.3f}")

# colour scaling: full cosine range [0, 1] so the colourbar reads naturally; fine blocks still
# glow and the near-chance order super-block stays dark (within-order 0.09 ~= global 0.05)
vmax = 1.0
disp = Sp.copy(); np.fill_diagonal(disp, np.nan)               # mask the unit diagonal

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.size": 7})
fig = plt.figure(figsize=(3.5, 3.72))
axm = fig.add_axes([0.034, 0.205, 0.964, 0.71])                 # fills the column width; strip left, colourbar above
# non-reversed magma: brightness == cosine (matches the caption). Fine blocks glow; the near-chance
# order region stays dark == background -- its darkness IS the honest finding, not a boosted signal
cmap = plt.get_cmap("magma").copy(); cmap.set_bad("#808080")
im = axm.imshow(disp, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest", aspect="auto")
axm.set_xticks([]); axm.set_yticks([])
# 12 order-boundary rules only (family lines dropped as clutter); sky-blue so they never read as data
for b in np.where(np.diff(o_s) != 0)[0] + 0.5:
    axm.axhline(b, color="#56B4E9", lw=0.6, alpha=0.9); axm.axvline(b, color="#56B4E9", lw=0.6, alpha=0.9)
axm.set_xlim(-0.5, N - 0.5); axm.set_ylim(N - 0.5, -0.5)

# Order margin strip (orientation only) along the left edge -- 13 maximally distinct SATURATED colours
# (tab20 pastels were hard to tell apart in a thin strip); majority order = dark navy
by_size = [i for i, _ in Counter(order).most_common()]; maj = by_size[0]
PALETTE = ["#0b2e63", "#e6194b", "#f58231", "#3cb44b", "#4363d8", "#911eb4", "#f032e6",
           "#469990", "#9a6324", "#800000", "#808000", "#ffd000", "#ff69b4"]
col = {o: PALETTE[k] for k, o in enumerate(by_size)}
idx_of = {o: j for j, o in enumerate(by_size)}
listed = ListedColormap([col[i] for i in by_size]); norm = BoundaryNorm(np.arange(len(by_size) + 1) - 0.5, len(by_size))
axs = fig.add_axes([0.006, 0.205, 0.026, 0.71])                # order strip flush to the left edge
axs.imshow(np.array([idx_of[o] for o in o_s]).reshape(-1, 1), aspect="auto", cmap=listed, norm=norm, interpolation="nearest")
axs.set_xticks([]); axs.set_yticks([])

# cosine-similarity colourbar at the TOP (horizontal, full width) -- kept clear of the bottom legend
cax = fig.add_axes([0.006, 0.918, 0.992, 0.016])               # colourbar spans the full column width (strip edge -> matrix edge)
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.ax.xaxis.set_ticks_position("top"); cb.ax.xaxis.set_label_position("top")
cb.set_label("cosine similarity", fontsize=7, labelpad=2)
cb.ax.tick_params(labelsize=6.5, pad=1.5)

# order legend: 4 rows -- three rows of 3 (longer names), evenly spread and aligned among themselves;
# + one row of 4 (the four shortest names) on its own even spread, NOT aligned to the 3-item rows
axl = fig.add_axes([0.0, 0.0, 1.0, 0.18]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
short4 = sorted(by_size, key=lambda i: (len(names[i]), by_size.index(i)))[:4]   # 4 shortest names
rest = [i for i in by_size if i not in short4]                                  # 9 longer, in size order
colx3 = [0.02, 0.36, 0.70]                                                     # 3-item rows: even 3-column spread
colx4 = [0.02, 0.27, 0.52, 0.77]                                              # bottom row: even, first box aligned to colx3[0]
rowy = [0.84, 0.60, 0.36, 0.12]                                                # 4 rows, top -> bottom
sw_w, sw_h = 0.0186, 0.13
def swatch(i, x, y):
    axl.add_patch(Rectangle((x, y - sw_h / 2), sw_w, sw_h, facecolor=col[i], edgecolor="#333",
                            lw=0.3, clip_on=False))
    axl.text(x + sw_w + 0.012, y, names[i], va="center", ha="left", fontsize=7.0)
for r in range(3):                                                             # rows 0-2: 3 items each
    for c in range(3):
        swatch(rest[r * 3 + c], colx3[c], rowy[r])
for c in range(4):                                                             # bottom row: 4 shortest
    swatch(short4[c], colx4[c], rowy[3])

for ext in ("pdf", "png"):
    out = str(_FIG / f"fig_simmatrix.{ext}")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.0, dpi=300)
    print("wrote", out)
