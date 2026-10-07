#!/usr/bin/env python3
"""Step 4 -- regenerate the manuscript's Figure 1, Figure 2, Table 1 and macro values (any OS).

Runs the two bundled generators against the artifacts in data/:
  * fig_sweep_table.py  -> figures/fig_pca_sweep.{pdf,png} + prints the Table 1 tabular and the
                           values of the paper's result macros (from experiments_out/*.json)
  * fig_simmatrix.py    -> figures/fig_simmatrix.{pdf,png} + prints the within-rank mean cosines
                           (from data/cub_species_centroids_4096.npy)

Needs only matplotlib/numpy/scipy/scikit-learn -- no GPU, no embeddings. This is what CI runs
to prove the paper's figures/numbers reproduce from the shipped data.

    python 04_make_figures.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent / "reprocub"


def main() -> int:
    argparse.ArgumentParser(description="Regenerate Figure 1, Figure 2, Table 1 and the paper's "
                            "result values from the bundled data/ (no options).").parse_args()
    for script in ("fig_sweep_table.py", "fig_simmatrix.py"):
        print(f"\n===== {script} =====", flush=True)
        subprocess.run([sys.executable, str(PKG / script)], check=True)
    print("\nOK -- figures in ./figures, macro/Table-1 values printed above.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
