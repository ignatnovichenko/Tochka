"""Read an explicitly selected OpenCode Go credential without copying it to disk."""
import json
import os
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import getpass
import warnings
from contextvars import ContextVar
from functools import wraps

GO_BASE_URL = 'https://opencode.ai/inference/go/openai/v1'
GO_KEY_ENV = 'OPENCODE_GO_API_KEY'
_session = ContextVar('tochka_credentials', default=None)


def interactive_credentials(function):
    """CLI-scoped cache, also shared by nested wizard -> run calls."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _session.get() is not None:
            return function(*args, **kwargs)
        cache = {}
        token = _session.set(cache)
        try:
            return function(*args, **kwargs)
        finally:
            cache.clear()
            _session.reset(token)
    return wrapped


def _identity(config):
    return (config.get('base_url', '').rstrip('/'), config.get('api_key_env', 'NIR_API_KEY'))


def request_api_key(config):
    """Prompt only in an interactive CLI execution; never fall back to echoed input."""
    value = api_key(config)
    if value:
        return value
    cache = _session.get()
    if cache is None or not sys.stdin.isatty():
        return ''
    value = hidden_key_input('API-ключ для ' + config['base_url'] + ' (ввод скрыт; сохранится на этом компьютере): ')
    from .key_store import save_key
    save_key(config, value)
    cache[_identity(config)] = value
    return value


def hidden_key_input(prompt):
    from .business.schema import ValidationError
    if not sys.stdin.isatty():
        raise ValidationError('Для скрытого ввода ключа откройте интерактивный терминал.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', getpass.GetPassWarning)
            value = getpass.getpass(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise ValidationError('Ввод API-ключа отменён. Запуск не выполнен.') from None
    except (getpass.GetPassWarning, OSError):
        raise ValidationError('Скрытый ввод недоступен. Запустите CLI в терминале или задайте переменную ключа.') from None
    if not value:
        raise ValidationError('API-ключ не введён. Запуск не выполнен.')
    return value


def saved_go_key():
    """Support the local credential DB and older auth.json; never log values."""
    root = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local/share') / 'opencode'
    db = root / 'opencode.db'
    if db.is_file():
        try:
            with closing(sqlite3.connect(db.as_uri() + '?mode=ro', uri=True, timeout=2)) as conn:
                rows = conn.execute(
                    'SELECT value FROM credential WHERE integration_id=? AND active=1',
                    ('opencode-go',),
                ).fetchall()
            # Ambiguous/missing active credentials must not select an arbitrary account.
            if len(rows) != 1:
                return ''
            value = json.loads(rows[0][0])
            key = value.get('key') if isinstance(value, dict) else None
            return key if isinstance(key, str) and key else ''
        except (sqlite3.Error, ValueError, TypeError, OSError):
            return ''
    try:
        value = json.loads((root / 'auth.json').read_text()).get('opencode-go', {})
        key = value.get('key') if isinstance(value, dict) and value.get('type') == 'api' else None
        return key if isinstance(key, str) and key else ''
    except (OSError, ValueError, AttributeError, TypeError):
        return ''


def api_key(config):
    cached = (_session.get() or {}).get(_identity(config))
    if cached:
        return cached
    env = config.get('api_key_env', 'NIR_API_KEY')
    value = os.environ.get(env, '')
    if value:
        return value
    from .key_store import stored_key
    value = stored_key(config)
    if value:
        return value
    # The saved credential is never supplied to any other endpoint/provider.
    if env == GO_KEY_ENV and config.get('base_url', '').rstrip('/') == GO_BASE_URL:
        return saved_go_key()
    return ''
