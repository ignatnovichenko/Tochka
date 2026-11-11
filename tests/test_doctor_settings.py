import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from nir_eval.doctor_settings import main, load_settings, save_settings, settings_path
from nir_eval.doctor import resolve_profile
from nir_eval.business.schema import ValidationError


class SettingsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        env = patch.dict(os.environ, {'XDG_CONFIG_HOME': temp.name}, clear=True)
        env.start(); self.addCleanup(env.stop)

    def test_configure_without_network_or_secret(self):
        messages = []
        self.assertEqual(main(['--model', 'fixture-model'], print_fn=messages.append), 0)
        profile = load_settings()
        self.assertEqual(profile['model'], 'fixture-model')
        self.assertNotIn('key', profile['provider'])
        self.assertEqual(resolve_profile(None, None), profile)

    def test_literal_key_field_rejected(self):
        profile = {'model': 'fixture', 'provider': {'type': 'openai-compatible', 'base_url': 'https://example.test/v1', 'key': 'fixture-secret'}}
        with self.assertRaises(ValidationError): save_settings(profile)
        self.assertFalse(settings_path().exists())

    def test_invalid_saved_settings_can_be_replaced(self):
        settings_path().parent.mkdir(parents=True)
        settings_path().write_text('invalid')
        with self.assertRaises(ValidationError): load_settings()
        self.assertEqual(main(['--model', 'new-model'], print_fn=lambda _: None), 0)
        self.assertEqual(load_settings()['model'], 'new-model')

    def test_symlink_settings_refused(self):
        p = settings_path(); p.parent.mkdir(parents=True)
        target = p.parent / 'other.json'; target.write_text('{}')
        p.symlink_to(target)
        with self.assertRaises(ValidationError): load_settings()
