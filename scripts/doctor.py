#!/usr/bin/env python3
"""Alias for python -m nir_eval doctor; no inference or network calls."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nir_eval.diagnostics import main
if __name__=='__main__':raise SystemExit(main())
