#!/bin/bash
set -uo pipefail
cd /workspace/shared/nir/mvp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/workspace/shared/nir/experiments/.venv/bin/python -m nir_eval business run --cases data/business_smoke_cases.jsonl --rubrics data/business_rubrics.jsonl --solver-config configs/business_solver_smoke.json --judge-config configs/business_judge_7b_smoke.json --output runs/business_neural_smoke_run02 --save-raw-judge
result=$?
python - "$result" <<'PY'
import datetime,json,pathlib,sys
p=pathlib.Path('/workspace/shared/nir/experiments/business_neural_smoke_run02/process_exit.json')
p.write_text(json.dumps({'exit_code':int(sys.argv[1]),'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
PY
exit "$result"
