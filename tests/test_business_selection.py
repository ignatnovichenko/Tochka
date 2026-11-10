"""Synthetic selector contract tests. No model calls; fixtures are not benchmark results."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from nir_eval.business_selection import select_business_run, validate_business_policy
from nir_eval.business.schema import ValidationError, digest


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.responses, self.solver, self.rows = [], [], []
        for config, score in [('a', .8), ('b', .5)]:
            for case in ['task1', 'task2']:
                key = {'id': case, 'model': 'test-solver', 'configuration': config, 'repetition': 0}
                text = 'Synthetic test answer'
                messages = [{'role': 'user', 'content': case}]
                self.responses.append({**key, 'provenance': 'model-inference', 'status': 'ok', 'response': text})
                self.solver.append({**key, 'provenance': 'model-inference', 'error': None,
                    'messages': messages, 'prompt_sha256': digest(messages), 'raw_output': text,
                    'latency_seconds': 1.0, 'latency_scope': 'synthetic Provider.generate call'})
                self.rows.append({**key, 'schema_version': 'business-judge-v1', 'checks_role': 'diagnostic_only',
                    'response_provenance': 'model-inference', 'response_status': 'ok',
                    'candidate_sha256': digest(text), 'case_sha256': digest(case),
                    'rubric_id': 'r1', 'rubric_sha256': digest('r1'),
                    'checks': [{'passed': False}], 'deterministic_status': 'failed',
                    'score': score, 'passed': score >= .75, 'status': 'passed' if score >= .75 else 'failed',
                    'judge': {'status': 'evaluated', 'model': 'synthetic-judge-fixture', 'provenance': 'model-inference',
                        'score': score, 'passed': score >= .75, 'critical_errors': [], 'latency_seconds': 2.0}})
        self.manifest = {'schema_version': 'business-pipeline-v1', 'state': 'completed',
            'target_quality_method': 'neural_judge_only', 'cases_sha256': digest('cases'),
            'rubrics_sha256': digest('rubrics'), 'solver_config_sha256': digest('solver'),
            'judge_config_sha256': digest('judge'), 'case_ids': ['task1', 'task2'],
            'configuration_names': ['a', 'b'], 'repetitions': 1,
            'planned_attempts': 4, 'solver_completed_attempts': 4, 'evaluated_attempts': 4,
            'solver_model': 'test-solver', 'judge_type': 'generation-provider', 'judge_evaluated': 4}
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        for filename, rows, field in [('responses.jsonl', self.responses, 'responses_sha256'),
             ('solver_records.jsonl', self.solver, 'solver_records_sha256'),
             ('records.jsonl', self.rows, 'records_sha256')]:
            path = self.root / filename
            path.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
            self.manifest[field] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))

    def select(self, policy=None):
        self.save()
        return select_business_run(self.root, policy or {})

    def unknown(self, index, status='unavailable'):
        row = self.rows[index]
        row.update(score=None, passed=None, status='evaluation_error' if status == 'error' else 'not_evaluated')
        row['judge'].update(score=None, passed=None, status=status)
        self.manifest['judge_evaluated'] -= 1

    def test_selects_neural_quality_not_diagnostics(self):
        result = self.select()
        self.assertEqual(result['status'], 'selected')
        self.assertEqual(result['selected']['configuration'], 'a')
        self.assertEqual(result['selected']['mean_judge_score'], .8)
        self.assertEqual(result['policy']['min_judge_coverage'], 1.0)
        self.assertFalse(result['model_calls_performed'])

    def test_all_unknown_is_never_zero(self):
        for i in range(4): self.unknown(i)
        result = self.select({'min_judge_coverage': 0})
        self.assertEqual(result['status'], 'no_eligible_configuration')
        self.assertTrue(all(c['mean_judge_score'] is None for c in result['candidates']))

    def test_partial_coverage_excludes_high_scoring_group_by_default(self):
        self.unknown(0)
        result = self.select()
        self.assertEqual(result['selected']['configuration'], 'b')
        a = result['candidates'][0]
        self.assertEqual(a['mean_judge_score'], .8)
        self.assertEqual(a['judge_coverage'], .5)
        self.assertIn('judge_coverage_below_threshold', a['rejection_reasons'])

    def test_explicit_partial_coverage_preserves_mean_and_unknown(self):
        self.unknown(0, 'error')
        result = self.select({'min_judge_coverage': .5})
        self.assertEqual(result['selected']['configuration'], 'a')
        self.assertEqual(result['selected']['mean_judge_score'], .8)
        self.assertEqual(result['selected']['judge_unknown'], 1)

    def test_genuine_zero_is_valid_score(self):
        for row in self.rows:
            row.update(score=0, passed=False, status='failed')
            row['judge'].update(score=0, passed=False)
        result = self.select()
        self.assertEqual(result['status'], 'tied_configurations')
        self.assertTrue(all(c['mean_judge_score'] == 0 for c in result['candidates']))

    def test_no_feasible_score(self):
        self.assertEqual(self.select({'min_mean_judge_score': .9})['status'], 'no_eligible_configuration')

    def test_tie_not_arbitrarily_selected(self):
        for row in self.rows[2:]:
            row.update(score=.8, passed=True, status='passed')
            row['judge'].update(score=.8, passed=True)
        result = self.select()
        self.assertEqual(result['status'], 'tied_configurations')
        self.assertIsNone(result['selected'])
        self.assertEqual(len(result['tied']), 2)

    def test_critical_neural_error_blocks_default_selection(self):
        self.rows[0].update(passed=False, status='failed')
        self.rows[0]['judge'].update(passed=False, critical_errors=[{'id': 'danger', 'detected': True}])
        result = self.select()
        self.assertEqual(result['selected']['configuration'], 'b')
        self.assertIn('neural_critical_error_rate_above_limit', result['candidates'][0]['rejection_reasons'])

    def test_explicit_critical_error_tolerance(self):
        self.rows[0].update(passed=False, status='failed')
        self.rows[0]['judge'].update(passed=False, critical_errors=[{'id': 'danger', 'detected': True}])
        self.assertEqual(self.select({'max_judge_critical_error_rate': .5})['selected']['configuration'], 'a')

    def test_critical_error_cannot_be_passed(self):
        self.rows[0]['judge']['critical_errors'] = [{'detected': True}]
        with self.assertRaises(ValidationError): self.select()

    def test_missing_cost_not_free(self):
        result = self.select({'max_mean_cost_usd': 0})
        self.assertEqual(result['status'], 'no_eligible_configuration')
        self.assertTrue(all('cost_unknown' in c['rejection_reasons'] for c in result['candidates']))

    def test_cost_needs_explicit_contract(self):
        for row in self.rows: row['total_cost_usd'] = 0
        self.assertEqual(self.select({'max_mean_cost_usd': 0})['status'], 'no_eligible_configuration')
        self.manifest['cost_measurement'] = {'currency': 'USD', 'scope': 'solver_plus_judge_per_attempt', 'basis': 'provider_reported'}
        self.assertEqual(self.select({'max_mean_cost_usd': 0})['status'], 'selected')

    def test_solver_latency_constraint(self):
        self.assertEqual(self.select({'max_mean_solver_latency': .5})['status'], 'no_eligible_configuration')
        self.assertEqual(self.select({'max_mean_solver_latency_seconds': 1})['status'], 'selected')

    def test_total_latency_needs_scope(self):
        result = self.select({'max_mean_total_latency_seconds': 5})
        self.assertEqual(result['status'], 'no_eligible_configuration')
        self.assertTrue(all('total_latency_unknown' in c['rejection_reasons'] for c in result['candidates']))
        self.manifest.update(judge_latency_scope='synthetic judge call', total_latency_scope='solver_plus_judge_call_seconds')
        result = self.select({'max_mean_total_latency': 3})
        self.assertEqual(result['selected']['mean_total_latency_seconds'], 3)

    def test_total_latency_incomplete_metrics(self):
        self.manifest.update(judge_latency_scope='judge call', total_latency_scope='solver_plus_judge_call_seconds')
        self.rows[0]['judge']['latency_seconds'] = None
        result = self.select({'max_mean_total_latency': 4})
        self.assertEqual(result['selected']['configuration'], 'b')
        self.assertIsNone(result['candidates'][0]['mean_total_latency_seconds'])

    def test_different_solver_scope_not_ranked_under_latency_constraint(self):
        for row in self.solver[2:]: row['latency_scope'] = 'different scope'
        result = self.select({'max_mean_solver_latency': 3})
        self.assertEqual(result['status'], 'no_eligible_configuration')
        self.assertEqual(self.select()['status'], 'selected')

    def test_hash_mismatch_fails_closed(self):
        with (self.root / 'records.jsonl').open('a') as stream: stream.write('\n')
        with self.assertRaisesRegex(ValidationError, 'hash mismatch'):
            select_business_run(self.root, {})

    def test_missing_matrix_row_rejected_even_if_counts_updated(self):
        self.responses.pop(); self.solver.pop(); self.rows.pop()
        for field in ('planned_attempts', 'solver_completed_attempts', 'evaluated_attempts', 'judge_evaluated'):
            self.manifest[field] = 3
        with self.assertRaises(ValidationError): self.select()

    def test_duplicate_attempt_rejected(self):
        self.responses.append(copy.deepcopy(self.responses[0]))
        with self.assertRaisesRegex(ValidationError, 'Duplicate'): self.select()

    def test_manifest_explicit_configuration_plan_checked(self):
        self.manifest['configuration_names'].append('absent')
        with self.assertRaises(ValidationError): self.select()

    def test_legacy_inferred_plan_has_warning(self):
        del self.manifest['configuration_names']; del self.manifest['repetitions']
        result = self.select()
        self.assertFalse(result['configuration_plan_explicit_in_manifest'])
        self.assertTrue(any('Legacy manifest' in x for x in result['warnings']))

    def test_running_or_interrupted_cannot_be_selected(self):
        for state in ('running', 'interrupted'):
            self.manifest['state'] = state
            with self.assertRaises(ValidationError): self.select()

    def test_replay_rejected(self):
        self.responses[0]['provenance'] = 'replay-not-inference'
        with self.assertRaises(ValidationError): self.select()

    def test_non_neural_evaluated_provenance_rejected(self):
        self.rows[0]['judge']['provenance'] = 'control'
        with self.assertRaises(ValidationError): self.select()

    def test_quality_cannot_be_replaced_with_code_score(self):
        self.rows[0]['score'] = .9
        with self.assertRaises(ValidationError): self.select()

    def test_unknown_score_cannot_be_imputed_zero(self):
        self.unknown(0); self.rows[0]['score'] = 0
        with self.assertRaises(ValidationError): self.select()

    def test_response_hash_corresponds_to_actual_answer(self):
        self.responses[0]['response'] = 'different'
        with self.assertRaises(ValidationError): self.select()

    def test_generation_error_is_not_quality_zero(self):
        self.responses[0].update(status='error', response='')
        self.solver[0].update(error='solver_provider_error', raw_output='')
        self.unknown(0, 'not_evaluated')
        self.rows[0].update(response_status='error', candidate_sha256=None)
        result = self.select({'min_judge_coverage': .5})
        a = result['candidates'][0]
        self.assertEqual(a['generation_error_rate'], .5)
        self.assertEqual(a['mean_judge_score'], .8)
        self.assertIn('generation_error_rate_above_limit', a['rejection_reasons'])

    def test_min_pass_rate_is_neural(self):
        result = self.select({'min_judge_pass_rate': 1})
        self.assertEqual(result['selected']['configuration'], 'a')

    def test_bad_policy_rejected(self):
        for policy in [{'min_judge_coverage': True}, {'min_judge_coverage': -1},
            {'min_mean_judge_score': float('nan')}, {'max_mean_cost_usd': -1},
            {'objective': 'latency'}, {'unexpected': 1}, {'min_tasks': True},
            {'max_mean_solver_latency': 1, 'max_mean_solver_latency_seconds': 1}]:
            with self.subTest(policy=policy), self.assertRaises(ValidationError):
                validate_business_policy(policy)


if __name__ == '__main__': unittest.main()
