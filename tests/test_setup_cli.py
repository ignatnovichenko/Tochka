"""Promptfoo setup UI contract; all installations are mocked."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from nir_eval import app, experiment, setup_cli, wizard
from nir_eval.business.schema import ValidationError
from nir_eval.interactive import run_shell


class SetupCliTests(unittest.TestCase):
    def setUp(self):
        self.lines = []

    def test_check_never_installs(self):
        for ready, code in [(True, 0), (False, 2)]:
            with self.subTest(ready=ready), patch('nir_eval.promptfoo_backend.runtime_status', return_value={'ready': ready}), \
                 patch('nir_eval.promptfoo_backend.setup_runtime') as install:
                self.assertEqual(setup_cli.main(['--check'], print_fn=self.lines.append), code)
                install.assert_not_called()

    def test_setup_installs_and_reports_readiness(self):
        with patch('nir_eval.promptfoo_backend.setup_runtime', return_value={'ready': True}) as install:
            self.assertEqual(setup_cli.main([], print_fn=self.lines.append), 0)
        install.assert_called_once_with()

    def test_failed_install_is_not_success(self):
        with patch('nir_eval.promptfoo_backend.setup_runtime', return_value={'ready': False}):
            self.assertEqual(setup_cli.main([], print_fn=self.lines.append), 2)
        with patch('nir_eval.promptfoo_backend.setup_runtime', side_effect=ValidationError('Node.js >=22.22.0')):
            self.assertEqual(setup_cli.main([], print_fn=self.lines.append), 2)
        self.assertIn('Node.js >=22.22.0', '\n'.join(self.lines))

    def test_app_dispatch(self):
        with patch('nir_eval.setup_cli.main', return_value=0) as command:
            self.assertEqual(app.main(['setup', '--check']), 0)
        command.assert_called_once_with(['--check'])

    def test_shell_has_setup_without_engine_choice(self):
        answers = iter(['/', '', '/setup --check', '/exit']); calls = []
        run_shell(input_fn=lambda _: next(answers), print_fn=self.lines.append,
                  command_fn=lambda args: calls.append(args) or 0)
        self.assertEqual(calls, [['setup', '--check']])
        self.assertTrue(any('/setup —' in s for s in self.lines))
        self.assertFalse(any(s.startswith('/engine —') for s in self.lines))

    def test_provider_always_promptfoo_and_engine_flag_removed(self):
        args = ['solver-add', 'folder', '--model', 'model', '--name', 'base', '--base-url', 'http://localhost:1234/v1']
        parsed = experiment.parser().parse_args(args)
        with patch('nir_eval.experiment.validate_api_config', return_value=True):
            provider, _ = experiment.configure_provider(parsed)
        self.assertEqual(provider['engine'], 'promptfoo')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiment.parser().parse_args(args + ['--engine', 'native'])

    def test_wizard_has_no_engine_option(self):
        settings = wizard.ModelSettings('base', 'model', 'http://localhost:1234/v1', 'NIR_API_KEY', 0, 100, None, None, '')
        self.assertNotIn('--engine', wizard.model_command('solver-add', Path('x'), settings, 1))

    def test_runtime_required_for_execution_even_without_engine_field(self):
        provider = {'type': 'openai-compatible'}
        judge = {'type': 'generation-provider', 'model': 'judge', 'provider': provider}
        solver = {'model': 'solver', 'provider': provider, 'configurations': [{'name': 'base'}]}
        with patch('nir_eval.experiment.config', return_value={'solvers':['s.json'],'judge':'j.json','policy':{}}), \
             patch('nir_eval.experiment.validate_business_policy'), \
             patch('nir_eval.experiment.import_inputs', return_value=([{}], {}, {})), \
             patch('nir_eval.experiment.load_judge_config', return_value=judge), \
             patch('nir_eval.business.running.load_solver_config', return_value=solver), \
             patch('nir_eval.experiment.validate_api_config'), \
             patch('nir_eval.promptfoo_backend.require_runtime') as require:
            experiment.experiment_plan(Path('x'), check_credentials=False)
            require.assert_not_called()
            experiment.experiment_plan(Path('x'), check_credentials=True)
            require.assert_called_once()
            require.side_effect = ValidationError('tochka setup')
            with self.assertRaisesRegex(ValidationError, 'tochka setup'):
                experiment.experiment_plan(Path('x'), check_credentials=True)


if __name__ == '__main__':
    unittest.main()
