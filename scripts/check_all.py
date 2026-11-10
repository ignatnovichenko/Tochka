#!/usr/bin/env python3
"""Reproduce lightweight checks without model inference or external network calls."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nir_eval.runner import load_config, run
from nir_eval.datasets import load_finqa
from nir_eval.scoring import execute_program, answers_equal


def main():
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT, check=True)
    cases = load_finqa(ROOT / 'data/finqa_dev.json')
    matches = sum(answers_equal(execute_program(c.expected_program, c.table), c.expected_answer) for c in cases)
    assert matches == len(cases)
    with tempfile.TemporaryDirectory(prefix='nir-check-') as directory:
        config = load_config(ROOT / 'configs/finqa_oracle.json')
        with redirect_stdout(io.StringIO()):
            oracle = run(config, Path(directory)/'oracle')
            config['provider']['type'] = 'invalid-control'
            invalid = run(config, Path(directory)/'invalid')
        assert all(g['answer_accuracy'] == 1 for g in oracle['groups'])
        assert all(g['answer_accuracy'] == 0 for g in invalid['groups'])
    print(json.dumps({'passed':True,'dev_reference_checks':matches,
                      'oracle_control':'all correct','invalid_control':'all rejected',
                      'model_inference_performed':False,'external_network_calls':0},indent=2))


if __name__ == '__main__': main()
