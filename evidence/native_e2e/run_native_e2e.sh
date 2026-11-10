#!/usr/bin/env bash
set -euo pipefail
cd /workspace/shared/nir/mvp
export HF_HOME=/workspace/shared/nir/experiments/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=/workspace/shared/nir/experiments/.venv/bin/python
"$PY" -m nir_eval run --config configs/e2e_qwen_direct.json --output runs/e2e_qwen_direct > ../experiments/native_e2e/direct_stdout.log 2>&1
"$PY" -m nir_eval run --config configs/e2e_qwen_cot.json --output runs/e2e_qwen_cot > ../experiments/native_e2e/cot_stdout.log 2>&1
"$PY" -m nir_eval compare runs/e2e_qwen_direct runs/e2e_qwen_cot --output runs/e2e_qwen_comparison > ../experiments/native_e2e/compare_stdout.json 2>&1
