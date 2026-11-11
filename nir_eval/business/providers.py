"""Separate, configurable judge transports. No credentials or HTTP bodies in errors."""
from __future__ import annotations
from dataclasses import dataclass, field
import os
import re
from typing import Protocol
import urllib.error
import urllib.parse
import urllib.request
from ..providers import ProviderUnavailable
from .schema import ValidationError, fields, string, number, strict_json, canonical


class JudgeUnavailable(RuntimeError):
    pass


class JudgeProviderError(RuntimeError):
    pass


@dataclass
class JudgeReply:
    text: str
    usage: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class JudgeProvider(Protocol):
    kind: str
    model: str
    def judge(self, messages: list[dict], case_id: str) -> JudgeReply: ...


class DisabledJudge:
    kind = 'disabled'
    model = 'not-connected'
    def judge(self, messages, case_id):
        raise JudgeUnavailable('judge_disabled')


class GenerationJudgeProvider:
    """Adapt existing generate(messages, model, parameters, context) providers."""
    def __init__(self, provider, model, parameters=None):
        self.provider = provider
        self.model = model
        self.parameters = parameters or {}
        self.kind = getattr(provider, 'kind', 'model-inference')

    def judge(self, messages, case_id):
        try:
            result = self.provider.generate(messages, self.model, self.parameters,
                                            {'id': case_id, 'config': 'business-judge', 'repetition': 0})
        except ProviderUnavailable as exc:
            if str(exc).startswith('api_') and re.fullmatch(r'api_[a-z0-9_]+', str(exc)):
                error = JudgeUnavailable('judge_' + str(exc))
                if hasattr(exc, 'retry_after_seconds'): error.retry_after_seconds = exc.retry_after_seconds
                raise error from None
            if str(exc) in {'opencode_free_limit', 'opencode_cooldown_active', 'opencode_timeout'}:
                raise JudgeUnavailable('judge_' + str(exc)) from None
            raise JudgeUnavailable('judge_connection_unavailable') from None
        except (TimeoutError, ConnectionError):
            raise JudgeUnavailable('judge_connection_unavailable') from None
        except Exception:
            raise JudgeProviderError('judge_provider_error') from None
        if not isinstance(result.text, str):
            raise JudgeProviderError('judge_response_not_text')
        return JudgeReply(result.text, result.usage, result.metadata)


class OpenAICompatibleJudge:
    """Direct judge settings use the same API-only transport and privacy contract."""
    kind = 'model-inference'

    def __init__(self, config, allow_remote=False):
        from ..providers import OpenAICompatibleProvider
        self.model = config['model']
        self.max_tokens = config.get('max_tokens', 2048)
        provider_config = {'type':'openai-compatible', 'base_url':config['base_url'],
                           'api_key_env':config.get('api_key_env','NIR_JUDGE_API_KEY'),
                           'timeout_seconds':config.get('timeout_seconds',90)}
        self.provider = OpenAICompatibleProvider(provider_config, allow_remote)

    def judge(self, messages, case_id):
        try:
            reply=self.provider.generate(messages,self.model,{'temperature':0,'max_tokens':self.max_tokens},{'id':case_id})
            return JudgeReply(reply.text,reply.usage,reply.metadata)
        except ProviderUnavailable as exc:
            code=str(exc)
            reason='judge_http_'+code.removeprefix('api_http_') if code.startswith('api_http_') else 'judge_connection_unavailable'
            error=JudgeUnavailable(reason)
            if hasattr(exc,'retry_after_seconds'):error.retry_after_seconds=exc.retry_after_seconds
            raise error from None
        except Exception:
            raise JudgeProviderError('judge_provider_error') from None


def validate_config(config):
    fields(config, ('type',), ('model', 'base_url', 'api_key_env', 'timeout_seconds', 'max_tokens', 'max_calls', 'provider', 'parameters', 'structured_retries'))
    kind = config['type']
    if not isinstance(kind, str) or kind not in {'disabled', 'openai-compatible', 'generation-provider'}:
        raise ValidationError('Unknown judge provider type')
    if kind == 'disabled':
        if set(config) != {'type'}:
            raise ValidationError('Disabled judge accepts no provider settings')
        return config
    string(config.get('model'), 200)
    if type(config.get('structured_retries', 0)) is not int or not 0 <= config.get('structured_retries', 0) <= 2:
        raise ValidationError('structured_retries must be 0, 1 or 2')
    if type(config.get('max_calls', 100)) is not int:
        raise ValidationError('Judge max_calls must be integer')
    number(config.get('max_calls', 100), 1, 1_000_000)
    if type(config.get('max_tokens', 2048)) is not int:
        raise ValidationError('Judge max_tokens must be integer')
    number(config.get('max_tokens', 2048), 64, 32_768)
    number(config.get('timeout_seconds', 90), .1, 3600)
    env = config.get('api_key_env', 'NIR_JUDGE_API_KEY')
    if not isinstance(env, str) or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,119}', env):
        raise ValidationError('Invalid API key environment variable name')
    if 'parameters' in config and (kind != 'generation-provider' or not isinstance(config['parameters'], dict)):
        raise ValidationError('Explicit parameters require a generation provider object')
    if 'parameters' in config and 'max_tokens' in config:
        raise ValidationError('Use either explicit parameters or max_tokens, not both')
    if kind == 'openai-compatible':
        string(config.get('base_url'), 2048)
        if 'provider' in config:
            raise ValidationError('Nested provider unsupported for HTTP judge')
    else:
        provider = config.get('provider')
        if not isinstance(provider, dict):
            raise ValidationError('generation-provider requires provider object')
        # Only a bounded explicit schema can enter persisted configuration.
        allowed = {'type', 'model_path', 'device', 'dtype', 'max_new_tokens', 'timeout_seconds',
                   'base_url', 'api_key_env', 'command', 'cwd', 'remote', 'revision', 'threads', 'seed', 'input_price_per_million', 'output_price_per_million', 'engine'}
        if provider.keys() - allowed or provider.get('type') not in {'openai-compatible', 'transformers-local', 'command', 'opencode'}:
            raise ValidationError('Unsupported nested generation provider configuration')
        if any(k in provider for k in {'base_url', 'api_key_env'}):
            if provider.get('type') != 'openai-compatible':
                raise ValidationError('Unexpected nested connection settings')
    return config


def _make_judge(config=None, allow_remote=False):
    config = validate_config(config or {'type': 'disabled'})
    if config['type'] == 'disabled':
        return DisabledJudge()
    if config['type'] == 'openai-compatible':
        return OpenAICompatibleJudge(config, allow_remote)
    # Optional integration with nir_eval's existing local/command adapters.
    from ..providers import make_provider
    provider = make_provider(config['provider'], allow_remote=allow_remote)
    parameters = ({'max_tokens': config.get('max_tokens', 2048), 'temperature': 0}
                  if config['provider']['type'] == 'openai-compatible'
                  else ({} if config['provider']['type'] == 'opencode' else {'max_new_tokens': config.get('max_tokens', 2048), 'do_sample': False}))
    if 'parameters' in config:
        parameters = config['parameters']
    return GenerationJudgeProvider(provider, config['model'], parameters)


class BudgetedJudge:
    """One shared budget counts initial calls AND explicit structured-output repairs."""
    def __init__(self, provider, max_calls, structured_retries):
        self.provider = provider
        self.parameters = getattr(provider, "parameters", {})
        self.model = provider.model
        self.kind = provider.kind
        self.remaining = max_calls
        self.structured_retries = structured_retries

    def judge(self, messages, case_id):
        if self.remaining <= 0:
            raise JudgeUnavailable('judge_call_budget_exhausted')
        self.remaining -= 1
        return self.provider.judge(messages, case_id)


def make_judge(config=None, allow_remote=False):
    config = validate_config(config or {'type': 'disabled'})
    provider = _make_judge(config, allow_remote)
    if config['type'] == 'disabled':
        return provider
    return BudgetedJudge(provider, config.get('max_calls', 100), config.get('structured_retries', 0))
