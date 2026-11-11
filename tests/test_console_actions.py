"""Guided menus exercise the same CLI and immutable run contracts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from nir_eval.console_actions import ConsoleActions
from nir_eval.interactive import run_shell
import test_api_workflow as api_fixture


class ConsoleActionTests(unittest.TestCase):
    def test_demo_creation_selects_experiment_and_keeps_path_spaces(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'demo with spaces'
            answers = iter(['/', '1', '2', str(path), '/exit'])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_shell(input_fn=lambda _: next(answers), print_fn=lambda _: None), 0)
            self.assertTrue((path / 'experiment.json').is_file())
            self.assertTrue((path / 'cases.jsonl').is_file())

    def test_global_keys_settings_work_without_an_experiment(self):
        answers = iter(['5'])
        calls = []
        actions = ConsoleActions(lambda _: next(answers), lambda _: None, lambda args: calls.append(args) or 0)
        self.assertEqual(actions.settings(), 0)
        self.assertEqual(calls, [['keys']])

    def test_invalid_experiment_does_not_dispatch_or_change_active(self):
        calls = []
        actions = ConsoleActions(lambda _: '', lambda _: None, lambda args: calls.append(args))
        actions.active = Path('/previous')
        self.assertEqual(actions.open_experiment(Path('/missing-experiment')), 2)
        self.assertEqual(actions.active, Path('/previous'))
        self.assertFalse(calls)

    def fixture(self):
        test = api_fixture.APIWorkflowTests()
        test.setUp()
        self.addCleanup(test.tearDown)
        test.prepare()
        return test

    def test_guided_api_run_results_reselect_and_resume(self):
        fixture = self.fixture()
        report = fixture.root / 'reselected'
        resumed = fixture.root / 'resumed'
        answers = iter([
            '/experiment', '3', str(fixture.experiment),
            '/settings', '4', '.8', '1', '', '', '.01', '0',
            '/run', '1', str(fixture.run), '2', 'да',
            '/results', '', 'да', str(report), '.8', '1', '', '', '0', '0',
            '/run', '2', str(fixture.run), str(resumed), '2', 'да',
            '/exit',
        ])
        output = []
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_shell(input_fn=lambda _: next(answers), print_fn=output.append), 0)
        self.assertEqual(len(fixture.requests), 16, 'Reselect and complete resume must make no extra API calls')
        self.assertEqual(fixture.read_comparison()['status'], 'selected')
        self.assertEqual(json.loads((report / 'comparison.json').read_text())['status'], 'no_eligible_configuration')
        self.assertEqual(fixture.read_comparison(resumed)['status'], 'selected')
        self.assertEqual(json.loads((fixture.experiment / 'experiment.json').read_text())['policy']['max_mean_cost_usd'], .01)
        self.assertTrue(any('API-вызовы' in line for line in output))

    def test_cancelled_run_has_no_model_calls_or_output(self):
        fixture = self.fixture()
        # Direct /experiment PATH is supported; enter it as one command.
        answers = iter([f'/experiment "{fixture.experiment}"', '/run', '1', str(fixture.run), '2', '', '/exit'])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_shell(input_fn=lambda _: next(answers), print_fn=lambda _: None), 0)
        self.assertEqual(fixture.requests, [])
        self.assertFalse(fixture.run.exists())


if __name__ == '__main__':
    unittest.main()
