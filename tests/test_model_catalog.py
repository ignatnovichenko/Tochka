import json
import unittest
from unittest.mock import MagicMock, patch

from nir_eval.model_catalog import fetch_models, choose_model, NoRedirect
from nir_eval.wizard import Questions, _model


class ModelCatalogTests(unittest.TestCase):
    def response(self, data):
        result = MagicMock()
        result.__enter__.return_value = result
        result.read.return_value = json.dumps(data).encode()
        return result

    def test_metadata_only_get_and_filtered_ids(self):
        opener = MagicMock()
        opener.open.return_value = self.response({'data': [{'id': 'model-b'}, {'id': 'model-a'}, {'id': 'model-a'}, {'id': 'sk-secret'}, {'id': 'bad\nterminal'}, {'id': 'image-x', 'type': 'image'}]})
        with patch('nir_eval.model_catalog.api_key', return_value='fixture-key'), patch('nir_eval.model_catalog.urllib.request.build_opener', return_value=opener):
            self.assertEqual(fetch_models('https://provider.example/v1', 'KEY'), ['model-a', 'model-b'])
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), 'GET')
        self.assertEqual(request.full_url, 'https://provider.example/v1/models')
        self.assertEqual(request.get_header('Authorization'), 'Bearer fixture-key')
        self.assertTrue(request.get_header('User-agent').startswith('Tochka/'))
        self.assertIsNone(request.data)

    def test_credential_redirect_is_refused(self):
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, '', {}, 'https://other.example'))

    def test_unavailable_catalog_returns_empty(self):
        opener = MagicMock()
        opener.open.side_effect = TimeoutError()
        with patch('nir_eval.model_catalog.api_key', return_value=''), patch('nir_eval.model_catalog.urllib.request.build_opener', return_value=opener):
            self.assertEqual(fetch_models('http://localhost:1234/v1', 'KEY'), [])

    def test_catalog_selection_cached_and_judge_excludes_solver(self):
        answers = iter(['1', '2', '1'])
        q = Questions(lambda _: next(answers), lambda _: None)
        with patch('nir_eval.model_catalog.fetch_models', return_value=['a', 'b', 'judge']) as fetch:
            self.assertEqual(choose_model(q, 'https://p.example/v1', 'KEY', _model), 'a')
            self.assertEqual(choose_model(q, 'https://p.example/v1', 'KEY', _model), 'b')
            self.assertEqual(choose_model(q, 'https://p.example/v1', 'KEY', _model, ['a', 'b']), 'judge')
        fetch.assert_called_once()

    def test_manual_fallback_is_available(self):
        q = Questions(lambda _: 'custom-model', lambda _: None)
        with patch('nir_eval.model_catalog.fetch_models', return_value=[]):
            self.assertEqual(choose_model(q, 'http://localhost/v1', 'KEY', _model), 'custom-model')

    def test_invalid_number_does_not_select_model(self):
        answers = iter(['99', '0', 'custom-model'])
        q = Questions(lambda _: next(answers), lambda _: None)
        with patch('nir_eval.model_catalog.fetch_models', return_value=['a']):
            self.assertEqual(choose_model(q, 'https://p.example/v1', 'KEY', _model), 'custom-model')


if __name__ == '__main__':
    unittest.main()
