#!/usr/bin/env bash
# Exact commands executed; retained for audit, not an idempotent replay script.
cd /workspace/shared/nir/mvp
export HF_HOME=/workspace/shared/nir/experiments/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/workspace/shared/nir/experiments/.venv/bin/python
# Completed native run, no import/replay:
"$PY" -m nir_eval run --config configs/e2e_qwen_direct.json --output runs/e2e_qwen_direct
# Original cot session lost transport after7saved records. Kept unchanged/excluded.
# A fresh complete native run used a NEW output directory, not import or merge:
"$PY" -m nir_eval run --config configs/e2e_qwen_cot.json --output runs/e2e_qwen_cot_recovered
"$PY" -m nir_eval compare runs/e2e_qwen_direct runs/e2e_qwen_cot_recovered --output runs/e2e_qwen_verified_comparison
# No inference here; frozen source snapshot independently recomputes saved scores:
"$PY" ../experiments/native_e2e/audit_native_e2e.py
