#!/bin/bash
set -euo pipefail
cd /workspace/shared/nir/mvp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/workspace/shared/nir/experiments/.venv/bin/python -m nir_eval business run --cases data/business_smoke_cases.jsonl --rubrics data/business_rubrics.jsonl --solver-config configs/business_solver_smoke.json --judge-config configs/business_judge_7b_smoke.json --output runs/business_neural_smoke --save-raw-judge
