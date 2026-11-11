import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nir_eval import app
from nir_eval.interactive import run_shell, command_parsers


class InteractiveTests(unittest.TestCase):
    def shell(self, entries, command=None):
        iterator = iter(entries)
        output, calls = [], []
        def ask(prompt):
            value = next(iterator, EOFError())
            if isinstance(value, BaseException):
                raise value
            return value
        status = run_shell(input_fn=ask, print_fn=output.append,
                           command_fn=command or (lambda args: calls.append(args) or 0))
        self.assertEqual(status, 0)
        return output, calls

    def test_menu_includes_every_experiment_command(self):
        output, calls = self.shell(['/', '', '/exit'])
        for name in command_parsers():
            self.assertTrue(any(('/'+name+' —') in line for line in output), name)
        self.assertFalse(calls)
        self.assertNotIn('/help', '\n'.join(output))

    def test_menu_selection_and_quoted_args(self):
        names = list(command_parsers())
        output, calls = self.shell(['/', str(names.index('init')+1), '"folder with spaces" --demo', '/exit'])
        self.assertEqual(calls, [['init', 'folder with spaces', '--demo']])
        self.assertTrue(any('/wizard' in s for s in output))

    def test_argument_help_keys_and_wizard_delegated(self):
        _, calls = self.shell(['/run --help', '/keys list', '/wizard', '/exit'])
        self.assertEqual(calls, [['run', '--help'], ['keys', 'list'], ['wizard']])

    def test_no_shell_execution(self):
        _, calls = self.shell(['/init "$(echo never); x" --demo', '/exit'])
        self.assertEqual(calls[0][1], '$(echo never); x')

    def test_failures_do_not_kill_shell_or_disclose_exception(self):
        def command(args):
            if args[0] == 'show':
                raise ValueError('sk-SECRET private customer data')
            if args[0] == 'run':
                raise SystemExit(2)
            if args[0] == 'plan':
                raise KeyboardInterrupt()
            raise SystemExit(0)
        output, _ = self.shell(['/show', '/run', '/plan', '/init --help', '/exit'], command)
        joined = '\n'.join(output)
        self.assertNotIn('SECRET', joined)
        self.assertIn('ValueError', joined)
        self.assertIn('прервана', joined)

    def test_eof_and_ctrl_c_at_prompt(self):
        output, _ = self.shell([KeyboardInterrupt(), EOFError()])
        self.assertIn('До встречи', '\n'.join(output))

    def test_active_experiment(self):
        with tempfile.TemporaryDirectory() as root:
            root = str(Path(root).resolve())
            Path(root, 'experiment.json').write_text('{}')
            _, calls = self.shell([f'/use "{root}"', '/show', '/models', '/run other --output out', '/exit'])
            self.assertEqual(calls, [['show', root], ['models', '--directory', root], ['run', 'other', '--output', 'out']])

    def test_invalid_use_does_not_dispatch(self):
        output, calls = self.shell(['/use /no/such/experiment', '/exit'])
        self.assertFalse(calls)
        self.assertIn('нет experiment.json', '\n'.join(output))

    def test_doctor_profile_active_path_and_safe_last_error(self):
        import sys
        from types import SimpleNamespace
        doctors, models = [], []
        profile = {'model': 'test', 'provider': {}}
        settings = SimpleNamespace(load_settings=lambda: profile,
                                   main=lambda args, **kw: models.append(args) or 0)
        doctor = SimpleNamespace(run_doctor=lambda **kw: doctors.append(kw) or 0)
        with patch.dict(sys.modules, {'nir_eval.doctor_settings': settings, 'nir_eval.doctor': doctor}):
            self.shell(['/run', '/model set', '/doctor', '/exit'], lambda a: 2)
        self.assertEqual(models, [['set']])
        self.assertEqual(doctors[0]['profile'], profile)
        self.assertIsNone(doctors[0]['experiment'])
        self.assertIn('кодом 2', doctors[0]['last_error'])

    def test_unknown_invalid_menu_and_bad_quotes_recover(self):
        output, calls = self.shell(['/help', '/wat', '/', '999', '/init "', '/exit'])
        self.assertFalse(calls)
        self.assertIn('Выберите номер', '\n'.join(output))

    def test_bare_tty_launches_shell(self):
        with patch('nir_eval.app.sys.stdin.isatty', return_value=True), patch('nir_eval.interactive.run_shell', return_value=0) as shell:
            self.assertEqual(app.main([]), 0)
            shell.assert_called_once_with()

    def test_bare_non_tty_help_without_reading(self):
        with patch('nir_eval.app.sys.stdin.isatty', return_value=False), patch('builtins.input', side_effect=AssertionError('must not read')), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(app.main([]), 0)
        self.assertIn('usage:', out.getvalue())
        self.assertIn('doctor', out.getvalue())


if __name__ == '__main__':
    unittest.main()
