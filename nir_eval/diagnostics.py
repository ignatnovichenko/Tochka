#!/usr/bin/env python3
"""Offline readiness check. Never downloads, calls models or sends user data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]


def diagnose():
    dataset=ROOT/'data/finqa_dev.json'
    source=ROOT/'data/finqa_dev.source.json'
    integrity=False
    if dataset.is_file() and source.is_file():
        integrity=hashlib.sha256(dataset.read_bytes()).hexdigest()==json.loads(source.read_text())['sha256']
    dependencies={name:importlib.util.find_spec(name) is not None
                  for name in ['openpyxl','torch','transformers','numpy','sympy']}
    return {'python':platform.python_version(),'interpreter':sys.executable,
            'core_ready':sys.version_info>=(3,10) and integrity,
            'bundled_finqa_sha_verified':integrity,'optional_dependencies':dependencies,
            'local_transformers_dependencies_available':dependencies['torch'] and dependencies['transformers'],
            'spreadsheet_dependency_available':dependencies['openpyxl'],
            'model_weights_in_release':False,'model_inference_performed':False,'network_calls':0,
            'llm_judge_validation_performed':False,
            'notes':['Missing optional dependencies do not block the standard-library FinQA core.',
                     'Presence of a library is not proof a model/server/key is configured.',
                     'Use the interpreter of your prepared virtual environment for local inference.',
                     'Saved reports can be audited without model weights or API access.']}


def main():
    result=diagnose();print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['core_ready'] else 1


if __name__=='__main__':raise SystemExit(main())
