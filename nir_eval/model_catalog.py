"""Read provider model metadata; never run inference or follow credential redirects."""
import json
import re
import urllib.error
import urllib.request
from http.client import HTTPException

from .api_support import validate_api_config
from .credentials import api_key, GO_BASE_URL
from . import __version__


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_models(base_url, key_env):
    config = {'base_url': base_url, 'api_key_env': key_env}
    local = validate_api_config(config, allow_remote=True)
    urls = [base_url.rstrip('/') + '/models']
    if base_url.rstrip('/') == GO_BASE_URL:
        urls.append('https://opencode.ai/inference/go/v1/models')
    key = api_key(config)
    headers = {'Accept': 'application/json', 'User-Agent': 'Tochka/' + __version__}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}) if local else urllib.request.ProxyHandler(), NoRedirect())
    for url in urls:
        try:
            request = urllib.request.Request(url, headers=headers)
            with opener.open(request, timeout=3) as response:
                raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                continue
            data = json.loads(raw)
            rows = data if isinstance(data, list) else data.get('data', data.get('models', []))
            if not isinstance(rows, list):
                continue
            result = set()
            for row in rows:
                if isinstance(row, dict) and row.get('type') in {'embedding', 'image', 'audio', 'video'}:
                    continue
                model = row.get('id') if isinstance(row, dict) else row
                if (isinstance(model, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}', model)
                        and not model.startswith(('sk-', 'ghp_', 'github_pat_')) and model != key):
                    result.add(model)
            if result:
                return sorted(result, key=str.casefold)
        except (OSError, HTTPException, ValueError, AttributeError, TypeError):
            continue
    return []


def choose_model(q, endpoint, key_env, validate, forbidden=()):
    if not hasattr(q, 'model_catalogs'):
        q.model_catalogs = {}
    identity = (endpoint, key_env)
    if identity not in q.model_catalogs:
        q.print('Загружаю список моделей API (без запросов к моделям)…')
        q.model_catalogs[identity] = fetch_models(endpoint, key_env)
    models = [model for model in q.model_catalogs[identity] if model not in forbidden]
    if not models:
        q.print('Каталог недоступен или пуст. Можно ввести ID вручную.')
        return q.ask('ID модели', validate=validate)
    q.print('Модели из каталога провайдера; доступность и лимиты проверяются при запуске:')
    for index, model in enumerate(models, 1):
        q.print(f'{index}. {model}')
    q.print('0. Ввести ID вручную')
    while True:
        answer = q.input('Выберите номер модели (или введите её ID): ').strip()
        if answer == '0':
            return q.ask('ID модели', validate=validate)
        if answer.isdigit():
            if not 1 <= int(answer) <= len(models):
                q.print('Выберите номер из списка.')
                continue
            answer = models[int(answer) - 1]
        try:
            return validate(answer)
        except ValueError as exc:
            q.print(str(exc))
