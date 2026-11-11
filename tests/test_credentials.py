"""Synthetic credentials only; no external inference."""
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from nir_eval.credentials import api_key, saved_go_key, GO_BASE_URL, GO_KEY_ENV, interactive_credentials, request_api_key
from nir_eval.api_support import validate_api_config
from nir_eval.business.schema import ValidationError
import getpass
import warnings
from nir_eval.providers import OpenAICompatibleProvider


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'opencode'
        self.root.mkdir()
        env = patch.dict(os.environ, {'XDG_DATA_HOME': self.temp.name, 'XDG_CONFIG_HOME': self.temp.name}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.config = {'type': 'openai-compatible', 'base_url': GO_BASE_URL, 'api_key_env': GO_KEY_ENV}

    def database(self, rows):
        with sqlite3.connect(self.root / 'opencode.db') as conn:
            conn.execute('CREATE TABLE credential(integration_id TEXT, value TEXT, active INTEGER)')
            conn.executemany('INSERT INTO credential VALUES (?,?,?)', rows)

    def test_active_go_only_and_no_disk_copy(self):
        self.database([('opencode', '{"key":"wrong"}', 1), ('opencode-go', '{"key":"fixture"}', 1)])
        db = self.root / 'opencode.db'
        before = db.read_bytes()
        with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('Unexpected model call')) as model_call:
            p = OpenAICompatibleProvider(self.config, allow_remote=True)
            self.assertEqual(p.api_key, 'fixture')
            model_call.assert_not_called()
        self.assertEqual(db.read_bytes(), before)
        self.assertFalse((self.root / 'auth.json').exists())
        self.assertNotIn(GO_KEY_ENV, os.environ)

    def test_no_saved_key_for_other_urls_or_env_names(self):
        with patch('nir_eval.credentials.saved_go_key', return_value='fixture') as read:
            for url in ['https://example.com/v1', GO_BASE_URL + '/other', GO_BASE_URL + '?x=1',
                        'https://opencode.ai.evil.test/inference/go/openai/v1']:
                self.assertEqual(api_key({**self.config, 'base_url': url}), '')
            self.assertEqual(api_key({**self.config, 'api_key_env': 'OTHER_KEY'}), '')
            read.assert_not_called()

    def test_explicit_env_takes_precedence(self):
        with patch.dict(os.environ, {GO_KEY_ENV: 'explicit'}), patch('nir_eval.credentials.saved_go_key') as read:
            self.assertEqual(api_key(self.config), 'explicit')
            read.assert_not_called()

    def test_inactive_not_used(self):
        self.database([('opencode-go', '{"key":"fixture"}', 0)])
        self.assertEqual(saved_go_key(), '')

    def test_ambiguous_not_used(self):
        self.database([('opencode-go', '{"key":"a"}', 1), ('opencode-go', '{"key":"b"}', 1)])
        self.assertEqual(saved_go_key(), '')

    def test_bad_value_not_used(self):
        self.database([('opencode-go', 'not-json', 1)])
        self.assertEqual(saved_go_key(), '')

    def test_old_auth_json_supported(self):
        (self.root / 'auth.json').write_text(json.dumps({'opencode-go': {'type': 'api', 'key': 'fixture'}}))
        self.assertEqual(saved_go_key(), 'fixture')

    def test_missing_is_empty(self):
        self.assertEqual(saved_go_key(), '')

    def test_prompt_once_and_persist_across_commands(self):
        @interactive_credentials
        def command():
            self.assertEqual(request_api_key(self.config), 'fixture-input')
            self.assertEqual(request_api_key(self.config), 'fixture-input')
            self.assertEqual(api_key(self.config), 'fixture-input')
            self.assertNotIn(GO_KEY_ENV, os.environ)
            self.assertEqual(api_key({**self.config, 'base_url': 'https://other.test/v1'}), '')
        with patch('sys.stdin.isatty', return_value=True), patch('getpass.getpass', return_value='fixture-input') as prompt:
            command()
            prompt.assert_called_once()
        self.assertEqual(api_key(self.config), 'fixture-input')
        self.assertEqual(list(self.root.iterdir()), [])

    def test_no_prompt_without_terminal(self):
        with patch('sys.stdin.isatty', return_value=False), patch('getpass.getpass') as prompt:
            self.assertEqual(interactive_credentials(lambda: request_api_key(self.config))(), '')
            prompt.assert_not_called()

    def test_cancel_empty_or_echo_fallback_aborts_and_clears(self):
        def echo_warning(*args):
            warnings.warn('echo fallback', getpass.GetPassWarning)
            self.fail('Echo fallback must not continue')
        for behavior in [EOFError(), KeyboardInterrupt(), echo_warning]:
            with self.subTest(behavior=type(behavior).__name__), patch('sys.stdin.isatty', return_value=True), patch('getpass.getpass', side_effect=behavior):
                with self.assertRaises(ValidationError):
                    interactive_credentials(lambda: request_api_key(self.config))()
                self.assertEqual(api_key(self.config), '')
        with patch('sys.stdin.isatty', return_value=True), patch('getpass.getpass', return_value=''):
            with self.assertRaises(ValidationError):
                interactive_credentials(lambda: request_api_key(self.config))()

    def test_preflight_only_prompts_after_explicit_remote_permission(self):
        @interactive_credentials
        def command():
            validate_api_config(self.config, allow_remote=True, require_key=False)
            with self.assertRaises(ValidationError):
                validate_api_config(self.config, allow_remote=False, require_key=True)
        with patch('sys.stdin.isatty', return_value=True), patch('getpass.getpass') as prompt:
            command()
            prompt.assert_not_called()

    def test_provider_uses_prompted_key_and_nested_session(self):
        @interactive_credentials
        def inner():
            return OpenAICompatibleProvider(self.config, allow_remote=True).api_key
        @interactive_credentials
        def outer():
            self.assertEqual(inner(), 'fixture-input')
            self.assertEqual(api_key(self.config), 'fixture-input')
        with patch('sys.stdin.isatty', return_value=True), patch('getpass.getpass', return_value='fixture-input'), \
                patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('Unexpected model call')) as model_call:
            outer()
            model_call.assert_not_called()
        self.assertEqual(api_key(self.config), 'fixture-input')
