"""Wizard interaction tests. These tests never invoke models or remote APIs."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from nir_eval.wizard import (ModelSettings, estimated_cost, main, model_command,
                             valid_endpoint, _env_name, Questions, _model_questions)


def answers(*, confirmation='', endpoint='', mode='1', import_paths=(), documents=(),
            judge_model='judge-model', judge_extra=(), prices=('', ''), tokens=(), concurrency=''):
    result = [mode, *import_paths, *[str(d) for d in documents], '', endpoint, '', '']
    for model in ['solver-a', 'solver-b']:
        result += [model, '', '', '', '', '', *prices]
    result += [judge_model, *judge_extra, '', '', '', '', *prices]
    result += ['', '', '', '', '', concurrency, *tokens, confirmation]
    return result


class WizardTests(unittest.TestCase):
    def setUp(self):
        catalog = patch('nir_eval.model_catalog.fetch_models', return_value=[])
        catalog.start()
        self.addCleanup(catalog.stop)
        self.go_patch = patch('nir_eval.credentials.saved_go_key', return_value='')
        self.go_patch.start()
        self.addCleanup(self.go_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        storage_patch = patch('nir_eval.key_store.store_path', return_value=Path(self.temp.name) / 'local-config' / 'credentials.json')
        storage_patch.start()
        self.addCleanup(storage_patch.stop)
        self.root = Path(self.temp.name).resolve() / 'settings'
        self.output = Path(self.temp.name).resolve() / 'results'
        self.commands = []
        self.messages = []

    def tearDown(self):
        self.temp.cleanup()

    def command(self, args):
        self.commands.append(args)
        # Exercise the real dataset initialization/import; stub only model setup/run.
        if args[0] in {'init', 'import'}:
            from nir_eval.experiment import main as experiment
            with contextlib.redirect_stdout(io.StringIO()):
                return experiment(args)
        return 0

    def run_dialog(self, supplied, command=None):
        stream = iter(supplied)
        def read(prompt):
            self.messages.append(prompt)
            try:
                return next(stream)
            except StopIteration:
                raise EOFError()
        code = main(['--directory', str(self.root), '--output', str(self.output)],
                    input_fn=read, print_fn=self.messages.append,
                    command_fn=command or self.command)
        self.assertEqual(list(stream), [], 'Unexpected unconsumed dialog answers')
        return code

    def test_default_saves_without_model_calls(self):
        self.assertEqual(self.run_dialog(answers()), 0)
        self.assertNotIn('run', [x[0] for x in self.commands])
        self.assertTrue((self.root / 'cases.jsonl').exists())
        self.assertIn('Оценка стоимости: неизвестна', '\n'.join(self.messages))
        solvers = [x for x in self.commands if x[0] == 'solver-add']
        self.assertEqual(len(solvers), 2)
        for command in solvers:
            self.assertEqual(command[command.index('--provider') + 1], 'api')
            self.assertEqual(command[command.index('--max-calls') + 1], '4')
        self.assertIn('Solver-запросов: 8', '\n'.join(self.messages))
        self.assertIn('Judge-запросов: 8', '\n'.join(self.messages))

    def test_only_explicit_yes_starts_local_run(self):
        self.assertEqual(self.run_dialog(answers(confirmation='да')), 0)
        self.assertEqual(self.commands[-1], ['run', str(self.root), '--output', str(self.output), '--concurrency', '4'])

    def test_requested_concurrency_is_in_run_and_saved_command(self):
        self.assertEqual(self.run_dialog(answers(confirmation='да', concurrency='2')), 0)
        self.assertEqual(self.commands[-1][-2:], ['--concurrency', '2'])
        self.assertIn('Одновременных запросов: до 2', '\n'.join(self.messages))

    def test_english_yes_does_not_start(self):
        self.assertEqual(self.run_dialog(answers(confirmation='yes')), 0)
        self.assertNotIn('run', [x[0] for x in self.commands])

    def test_remote_missing_key_cannot_start(self):
        with patch.dict(os.environ, {}, clear=True), patch('sys.stdin.isatty', return_value=False):
            self.assertEqual(self.run_dialog(answers(confirmation='да', endpoint='https://api.example/v1')), 2)
        self.assertNotIn('run', [x[0] for x in self.commands])
        self.assertTrue((self.root / 'cases.jsonl').exists())
        self.assertIn('NIR_API_KEY', '\n'.join(self.messages))

    def test_remote_hidden_prompt_only_after_confirmation_and_not_saved(self):
        secret = 'fixture-hidden-input'
        with patch.dict(os.environ, {}, clear=True), patch('sys.stdin.isatty', return_value=True), patch('getpass.getpass', return_value=secret) as prompt:
            self.assertEqual(self.run_dialog(answers(confirmation='да', endpoint='https://api.example/v1')), 0)
            prompt.assert_called_once()
            self.assertNotIn('NIR_API_KEY', os.environ)
        self.assertIn('run', [c[0] for c in self.commands])
        self.assertNotIn(secret, '\n'.join(self.messages) + json.dumps(self.commands))
        for path in self.root.rglob('*'):
            if path.is_file():
                self.assertNotIn(secret, path.read_text())

    def test_save_only_does_not_ask_for_key(self):
        with patch.dict(os.environ, {}, clear=True), patch('sys.stdin.isatty', return_value=True), patch('getpass.getpass') as prompt:
            self.assertEqual(self.run_dialog(answers(endpoint='https://api.example/v1')), 0)
            prompt.assert_not_called()

    def test_remote_consent_and_secret_never_persisted_or_printed(self):
        secret = 'do-not-print-this-key'
        with patch.dict(os.environ, {'NIR_API_KEY': secret}):
            self.assertEqual(self.run_dialog(answers(confirmation='да', endpoint='https://api.example/v1')), 0)
        self.assertIn('--allow-remote', self.commands[-1])
        self.assertNotIn(secret, '\n'.join(self.messages) + json.dumps(self.commands))
        self.assertIn('внешние API', '\n'.join(self.messages))

    def test_independent_judge_is_enforced(self):
        self.assertEqual(self.run_dialog(answers(judge_model='solver-a', judge_extra=['judge-model'])), 0)
        judges = [x for x in self.commands if x[0] == 'judge-set']
        self.assertEqual(judges[0][judges[0].index('--model') + 1], 'judge-model')
        self.assertIn('Судья должен отличаться', '\n'.join(self.messages))

    def test_existing_directory_does_not_overwrite(self):
        self.root.mkdir()
        sentinel = self.root / 'data.txt'
        sentinel.write_text('preserve')
        self.assertEqual(main(['--directory', str(self.root)], input_fn=lambda _: self.fail('Should not prompt'),
                              print_fn=self.messages.append, command_fn=self.command), 2)
        self.assertEqual(sentinel.read_text(), 'preserve')
        self.assertEqual(self.commands, [])

    def test_same_or_nested_output_rejected_before_setup(self):
        for output in [self.root, self.root / 'results', self.root.parent]:
            self.assertEqual(main(['--directory', str(self.root), '--output', str(output)],
                                  input_fn=lambda _: self.fail('Should reject paths before prompting'),
                                  print_fn=self.messages.append, command_fn=self.command), 2)
        self.assertEqual(self.commands, [])

    def test_eof_before_confirmation_has_no_run(self):
        supplied = answers()[:-1]
        self.assertEqual(self.run_dialog(supplied), 130)
        self.assertNotIn('run', [x[0] for x in self.commands])

    def test_import_and_shared_document(self):
        from nir_eval import wizard
        source = Path(wizard.__file__).parent / 'demo'
        cases = Path(self.temp.name) / 'import-cases.jsonl'
        rubrics = Path(self.temp.name) / 'import-rubrics.jsonl'
        shutil.copyfile(source / 'cases.jsonl', cases)
        shutil.copyfile(source / 'rubrics.jsonl', rubrics)
        doc = Path(self.temp.name) / 'rules.txt'
        doc.write_text('Общие правила компании.', encoding='utf-8')
        original = cases.read_bytes()
        self.assertEqual(self.run_dialog(answers(mode='2', import_paths=[str(cases), str(rubrics)], documents=[doc])), 0)
        saved = [json.loads(x) for x in (self.root / 'cases.jsonl').read_text().splitlines()]
        self.assertTrue(all('Общие правила компании.' in x['context'] for x in saved))
        self.assertEqual(cases.read_bytes(), original)

    def test_known_prices_and_manual_token_scenario(self):
        self.assertEqual(self.run_dialog(answers(prices=('1', '2'), tokens=('1000', '2000'))), 0)
        self.assertIn('Это не гарантия и не денежный бюджет', '\n'.join(self.messages))
        for command in self.commands:
            if command[0] in {'solver-add', 'judge-set'}:
                self.assertIn('--input-price-per-million', command)
                self.assertIn('--output-price-per-million', command)

    def test_failure_during_setup_stops_before_run(self):
        def command(args):
            result = self.command(args)
            return 2 if args[0] == 'solver-add' else result
        # Only confirmation remains unconsumed when configuration fails early.
        stream = iter(answers(confirmation='да'))
        code = main(['--directory', str(self.root), '--output', str(self.output)],
                    input_fn=lambda _: next(stream), print_fn=self.messages.append, command_fn=command)
        self.assertEqual(code, 2)
        self.assertNotIn('run', [x[0] for x in self.commands])


class WizardValidationTests(unittest.TestCase):
    def test_judge_token_default_and_explicit_override(self):
        for token_input, expected in [('',8192),('3072',3072)]:
            replies=iter(['judge','','','',token_input,'',''])
            q=Questions(lambda _:next(replies),lambda _:None)
            settings=_model_questions(q,'judge','http://localhost/v1','KEY',judge=True)
            self.assertEqual(settings.max_tokens,expected)

    def test_tariffs_must_be_complete_pair(self):
        replies = iter(['solver', '', '', '', '', '', '1', '', '', ''])
        messages = []
        q = Questions(lambda _: next(replies), messages.append)
        model = _model_questions(q, 'a', 'http://localhost/v1', 'KEY')
        self.assertIsNone(model.input_price)
        self.assertIsNone(model.output_price)
        self.assertTrue(any('оба тарифа' in message for message in messages))

    def test_api_endpoint_security(self):
        self.assertEqual(valid_endpoint('http://localhost:8000/v1/'), 'http://localhost:8000/v1')
        for value in ['http://external.example/v1', 'https://user:secret@example/v1',
                      'https://example/v1?api_key=secret', 'https://example/v1#secret', 'file:///tmp/model']:
            with self.assertRaises(ValueError): valid_endpoint(value)

    def test_keys_are_names_only(self):
        self.assertEqual(_env_name('COMPANY_API_KEY'), 'COMPANY_API_KEY')
        for value in ['sk-abcdef', 'ghp_abcdef', 'not a name', '']:
            with self.assertRaises(ValueError): _env_name(value)

    def test_cost_missing_prices_stays_unknown(self):
        a = ModelSettings('a', 'a', 'http://localhost/v1', 'KEY', 0, 100, 0, None)
        b = ModelSettings('b', 'b', 'http://localhost/v1', 'KEY', 0, 100, 0, 0)
        self.assertIsNone(estimated_cost([a], b, 4, 1, 1000, 1000))

    def test_cost_accounts_for_judge_repairs(self):
        a = ModelSettings('a', 'a', 'http://localhost/v1', 'KEY', 0, 100, 1, 2)
        b = ModelSettings('b', 'b', 'http://localhost/v1', 'KEY', 0, 200, 1, 2)
        self.assertAlmostEqual(estimated_cost([a], b, 4, 1, 1000, 2000), 4 * .0012 + 8 * .0024)
        self.assertIsNone(estimated_cost([a], b, 4, 1))

    def test_empty_tariffs_are_not_implicitly_zero(self):
        a = ModelSettings('a', 'a', 'http://localhost/v1', 'KEY', 0, 100)
        self.assertIsNone(estimated_cost([a], a, 4, 0))
        a.input_price = a.output_price = 0
        self.assertEqual(estimated_cost([a], a, 4, 0), 0)


if __name__ == '__main__':
    unittest.main()
