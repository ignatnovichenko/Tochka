#!/usr/bin/env python3
"""Reproducible non-LLM controls on the first N real benchmark instructions."""
import argparse
from pathlib import Path
import shutil
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nir_eval.spreadsheet import DEFAULT_DATASET, evaluate, inventory, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--tasks', type=int, default=10)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'runs/sheet_smoke')
    args = parser.parse_args()
    if args.out_dir.resolve().is_relative_to(args.dataset_root.resolve()):
        parser.error('--out-dir must be outside the dataset root')
    if args.tasks < 1:
        parser.error('--tasks must be positive')
    manifest = inventory(args.dataset_root)
    selected = manifest['tasks'][:args.tasks]
    task_ids = [task['id'] for task in selected]
    summaries = {}
    with tempfile.TemporaryDirectory(prefix='spreadsheetbench_controls_') as temp:
        for control, source_key, policy in [
            ('gold_copy_cached', 'reference_path', 'cached'),
            ('gold_copy_strict', 'reference_path', 'strict'),
            ('input_copy_noop', 'input_path', 'strict'),
        ]:
            predictions = Path(temp) / control
            for task in selected:
                destination = predictions / task['id']
                destination.mkdir(parents=True)
                for case in task['cases']:
                    source = case[source_key]
                    if source:
                        shutil.copyfile(source, destination / f"{case['case_index']}_{task['id']}_output.xlsx")
            report = evaluate(args.dataset_root, predictions, provenance='control-not-llm',
                              task_ids=task_ids, formula_policy=policy,
                              run_metadata={'control': control, 'model_invoked': False,
                                  'selection': 'first N instructions in dataset.json; not random or representative',
                                  'prediction_artifacts': 'temporary copies, deleted after control run',
                                  'recalculation_performed': False})
            write_json(args.out_dir / f'smoke_{control}.json', report)
            summaries[control] = report['summary']
            print(control, report['summary'], flush=True)
    write_json(args.out_dir / 'smoke_summary.json', {
        'kind': 'evaluator_controls_not_llm_results', 'selected_task_ids': task_ids,
        'model_invoked': False, 'controls': summaries})


if __name__ == '__main__':
    main()
