import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nir_eval.credentials import api_key, GO_BASE_URL, GO_KEY_ENV
from nir_eval.key_store import save_key, delete_key, read_store, store_path
from nir_eval.keys_cli import main
from nir_eval.business.schema import ValidationError


class KeyStoreTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        env = patch.dict(os.environ, {'XDG_CONFIG_HOME': temp.name, 'XDG_DATA_HOME': temp.name}, clear=True)
        env.start(); self.addCleanup(env.stop)
        self.config = {'base_url': GO_BASE_URL, 'api_key_env': GO_KEY_ENV}
        self.secret = 'fixture-secret-12345'

    def call(self, args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = main(args)
        return result, out.getvalue()

    def test_persist_scoped_permissions_and_delete(self):
        save_key(self.config, self.secret, 'go')
        self.assertEqual(api_key(self.config), self.secret)
        self.assertEqual(store_path().stat().st_mode & 0o777, 0o600)
        self.assertEqual(store_path().parent.stat().st_mode & 0o777, 0o700)
        self.assertEqual(api_key({**self.config, 'base_url': 'https://other.test/v1'}), '')
        self.assertEqual(api_key({**self.config, 'api_key_env': 'OTHER'}), '')
        delete_key('go')
        self.assertEqual(api_key(self.config), '')

    def test_add_list_show_delete(self):
        with patch('sys.stdin.isatty', return_value=True), patch('getpass.getpass', return_value=self.secret):
            code, output = self.call(['add', 'go'])
        self.assertEqual(code, 0)
        self.assertNotIn(self.secret, output)
        code, output = self.call(['list'])
        self.assertIn('go |', output)
        self.assertIn('********', output)
        self.assertNotIn(self.secret, output)
        self.assertEqual(self.call(['show', 'go'])[0], 2)
        with patch('sys.stdout.isatty', return_value=True):
            # redirect_stdout installs another stream, so use a stream with tty explicitly.
            out = io.StringIO(); out.isatty = lambda: True
            with contextlib.redirect_stdout(out): self.assertEqual(main(['show', 'go']), 0)
            self.assertIn(self.secret, out.getvalue())
        self.assertEqual(self.call(['delete', 'go'])[0], 0)
        self.assertEqual(read_store(), {})

    def test_menu_add_and_delete(self):
        with patch('sys.stdin.isatty', return_value=True), patch('getpass.getpass', return_value=self.secret), patch('builtins.input', side_effect=['1', 'go', '', '', '2', 'go', '']):
            code, output = self.call([])
        self.assertEqual(code, 0)
        self.assertEqual(read_store(), {})
        self.assertNotIn(self.secret, output)

    def test_overwrite_explicit_and_other_keys_preserved(self):
        save_key(self.config, self.secret, 'go')
        other = {'base_url': 'https://other.test/v1', 'api_key_env': 'OTHER'}
        save_key(other, 'other-fixture', 'other')
        with self.assertRaises(ValidationError): save_key(self.config, 'new-fixture', 'go')
        save_key(self.config, 'new-fixture', 'go', replace=True)
        self.assertEqual(api_key(self.config), 'new-fixture')
        self.assertEqual(api_key(other), 'other-fixture')

    def test_corrupt_store_never_overwritten_or_echoed(self):
        save_key(self.config, self.secret, 'go')
        store_path().write_text('broken ' + self.secret)
        before = store_path().read_bytes()
        code, output = self.call(['list'])
        self.assertEqual(code, 2)
        self.assertNotIn(self.secret, output)
        with self.assertRaises(ValidationError): save_key(self.config, 'new', 'new')
        self.assertEqual(before, store_path().read_bytes())

    def test_insecure_file_and_symlink_rejected(self):
        save_key(self.config, self.secret, 'go')
        store_path().chmod(0o644)
        with self.assertRaises(ValidationError): read_store()
        store_path().chmod(0o600)
        target = self.root / 'external.json'
        store_path().rename(target)
        store_path().symlink_to(target)
        with self.assertRaises(ValidationError): read_store()

    def test_invalid_endpoint_never_prompts(self):
        with patch('getpass.getpass') as prompt:
            code, _ = self.call(['add', 'bad', '--base-url', 'https://user:secret@example.test/v1'])
            self.assertEqual(code, 2)
            prompt.assert_not_called()

    def test_cli_entry_and_help(self):
        from nir_eval.app import main as app
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(app(['keys', 'list']), 0)
        self.assertIn('Сохранённых ключей пока нет', out.getvalue())
