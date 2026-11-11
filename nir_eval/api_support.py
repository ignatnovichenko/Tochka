"""OpenAI-compatible transport contracts and explicit, non-imputed costs."""
import math
import re
from urllib.parse import urlsplit
from .business.schema import ValidationError


def validate_api_config(config, *, allow_remote=False, require_key=False):
    allowed = {'type', 'base_url', 'api_key_env', 'timeout_seconds',
               'input_price_per_million', 'output_price_per_million'}
    if not isinstance(config, dict) or config.keys() - allowed:
        raise ValidationError('API config accepts URL, key environment name, timeout and optional tariffs only')
    base = config.get('base_url', '')
    if not isinstance(base, str):
        raise ValidationError('Invalid API URL')
    parsed = urlsplit(base)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
        raise ValidationError('Specify an HTTP(S) API base URL')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValidationError('API URL must not contain credentials, query or fragment')
    local = parsed.hostname in {'localhost', '127.0.0.1', '::1'}
    if not local and parsed.scheme != 'https':
        raise ValidationError('External API requires HTTPS')
    if not local and not allow_remote:
        raise ValidationError('External API requires explicit --allow-remote')
    env = config.get('api_key_env', 'NIR_API_KEY')
    if not isinstance(env, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,119}', env):
        raise ValidationError('Invalid API key environment variable name')
    timeout = config.get('timeout_seconds', 180)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValidationError('API timeout must be between 0 and 3600 seconds')
    prices = [config.get(k) for k in ('input_price_per_million','output_price_per_million')]
    if any(x is not None for x in prices):
        if any(type(x) not in (int,float) or not math.isfinite(x) or x < 0 for x in prices):
            raise ValidationError('Specify both non-negative input/output prices, or neither')
    if not local and require_key:
        from .credentials import request_api_key
        if not request_api_key(config):
            raise ValidationError('API-ключ отсутствует: запустите CLI в терминале для скрытого ввода или задайте переменную ' + env)
    return local


def response_metadata(data, config):
    if not isinstance(data,dict) or not isinstance(data.get('choices'),list) or not data['choices'] or not isinstance(data['choices'][0],dict):
        raise ValueError('Malformed API choices')
    choice = data['choices'][0]
    usage = data.get('usage') if isinstance(data.get('usage'), dict) else {}
    meta = {'response_model': data.get('model') if isinstance(data.get('model'),str) else None, 'finish_reason': choice.get('finish_reason') if isinstance(choice.get('finish_reason'),str) else None,
            'transport': 'openai-compatible-api', 'cost_usd': None, 'cost_basis': 'unknown'}
    opencode = usage.get('opencode') if isinstance(usage.get('opencode'), dict) else {}
    cost = usage.get('cost')
    finite = lambda x: type(x) in (int,float) and math.isfinite(x) and x >= 0
    if finite(cost):
        if opencode.get('estimated') is True:
            meta.update(estimated_cost_usd=cost, cost_basis='provider_estimate')
        else:
            meta.update(cost_usd=cost, cost_basis='provider_reported')
    elif all(type(config.get(k)) in (int,float) and math.isfinite(config[k]) and config[k]>=0 for k in ('input_price_per_million','output_price_per_million')):
        inp, out = usage.get('prompt_tokens'), usage.get('completion_tokens')
        if type(inp) is int and type(out) is int and inp >= 0 and out >= 0:
            estimate=(inp*config['input_price_per_million']+out*config['output_price_per_million'])/1_000_000
            if math.isfinite(estimate):
                meta.update(estimated_cost_usd=estimate,cost_basis='configured_tariff_estimate')
    if opencode.get('billing_source') in ('free','balance','go','go-plus','go-teams','byok'):
        meta['billing_source'] = opencode['billing_source']
    if finite(opencode.get('list_cost')):
        meta['list_cost_usd'] = opencode['list_cost']
    # Keep native numerical categories, never arbitrary provider strings/secrets.
    native = opencode.get('tokens')
    if isinstance(native, dict):
        meta['native_tokens'] = {k:v for k,v in native.items() if k in {'input','output','reasoning','cache_read','cache_write_5m','cache_write_1h'} and type(v) is int and v >= 0}
    return meta
