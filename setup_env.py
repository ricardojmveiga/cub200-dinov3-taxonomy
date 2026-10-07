#!/usr/bin/env python3
"""Create a virtual environment and install the dependencies -- one command, any OS.

Uses the stdlib :mod:`venv` and :mod:`pathlib`, so it behaves identically on Windows
(``.venv\\Scripts``), macOS and Linux (``.venv/bin``). No shell scripts, no activation
needed -- it prints the exact commands to run afterwards.

    python setup_env.py                 # core deps (CPU: experiments, figures, tests)
    python setup_env.py --with-torch    # also install torch/transformers (extraction, GPU/MPS)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def venv_python(vdir: Path) -> Path:
    # Windows puts the interpreter in Scripts\, POSIX in bin/
    return vdir / ("Scripts" if sys.platform.startswith("win") else "bin") / \
        ("python.exe" if sys.platform.startswith("win") else "python")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-torch", action="store_true",
                    help="also install torch/transformers/pillow (extraction + GPU/MPS)")
    ap.add_argument("--venv", type=Path, default=VENV)
    args = ap.parse_args()

    if not venv_python(args.venv).exists():
        print(f"[setup] creating venv at {args.venv} ...", flush=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(args.venv)
    py = venv_python(args.venv)

    def pip(*a):
        subprocess.run([str(py), "-m", "pip", *a], check=True)

    pip("install", "--upgrade", "pip")
    pip("install", "-r", str(ROOT / "requirements.txt"))
    if args.with_torch:
        print("[setup] installing torch/transformers (choose the CUDA wheel yourself for a GPU)", flush=True)
        pip("install", "-r", str(ROOT / "requirements-torch.txt"))

    activate = (args.venv / "Scripts" / "activate") if sys.platform.startswith("win") \
        else (args.venv / "bin" / "activate")
    print("\n[setup] done. Next:")
    print(f"  {py} -m pytest tests/           # verify")
    print(f"  {py} 04_make_figures.py         # reproduce Figure 1/2 + Table 1")
    print(f"  (or activate: {activate})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
