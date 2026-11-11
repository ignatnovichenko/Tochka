"""Local plaintext credential store, protected with owner-only POSIX permissions."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from .business.schema import ValidationError


def store_path():
    return Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config') / 'tochka' / 'credentials.json'


def identity(config):
    return (config['base_url'].rstrip('/'), config.get('api_key_env', 'NIR_API_KEY'))


def _validate_entry(entry):
    from .api_support import validate_api_config
    if not isinstance(entry, dict) or set(entry) != {'base_url', 'api_key_env', 'key'}:
        raise ValueError()
    validate_api_config({k: v for k, v in entry.items() if k != 'key'}, allow_remote=True)
    key = entry['key']
    if not isinstance(key, str) or not key or any(ord(c) < 33 or ord(c) == 127 for c in key):
        raise ValueError()


def read_store():
    path = store_path()
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    except FileNotFoundError:
        return {}
    except OSError:
        raise ValidationError('Не удалось открыть локальное хранилище ключей.') from None
    try:
        with os.fdopen(fd, 'r', encoding='utf-8') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or (os.name == 'posix' and (info.st_mode & 0o077 or info.st_uid != os.getuid())):
                raise ValueError()
            data = json.load(stream)
        if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('keys'), dict):
            raise ValueError()
        seen = set()
        for name, entry in data['keys'].items():
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name):
                raise ValueError()
            _validate_entry(entry)
            ident = identity(entry)
            if ident in seen:
                raise ValueError()
            seen.add(ident)
        return data['keys']
    except (ValueError, TypeError, OSError):
        raise ValidationError('Хранилище ключей повреждено или имеет небезопасные права доступа. Оно не перезаписано.') from None


@contextmanager
def _locked():
    import fcntl
    root = store_path().parent
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or root.stat().st_uid != os.getuid():
        raise ValidationError('Небезопасная папка хранилища ключей.')
    root.chmod(0o700)
    fd = os.open(root / '.credentials.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _write(entries):
    path = store_path()
    fd, name = tempfile.mkstemp(prefix='.credentials-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump({'version': 1, 'keys': entries}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_key(config, key, name=None, replace=False):
    entry = {'base_url': config['base_url'].rstrip('/'), 'api_key_env': config.get('api_key_env', 'NIR_API_KEY'), 'key': key}
    try:
        _validate_entry(entry)
    except (ValueError, TypeError):
        raise ValidationError('Некорректный адрес API, имя переменной или ключ.') from None
    ident = identity(entry)
    if name is None:
        name = 'api-' + hashlib.sha256(json.dumps(ident).encode()).hexdigest()[:16]
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name):
        raise ValidationError('Имя ключа: 1–80 латинских букв, цифр, дефисов или подчёркиваний.')
    with _locked():
        entries = read_store()
        conflicts = [n for n, e in entries.items() if n == name or identity(e) == ident]
        if conflicts and not replace:
            raise ValidationError('Такое имя или подключение уже сохранено. Для замены используйте --replace.')
        for n in conflicts:
            del entries[n]
        entries[name] = entry
        _write(entries)
    return name


def stored_key(config):
    ident = identity(config)
    return next((e['key'] for e in read_store().values() if identity(e) == ident), '')


def delete_key(name):
    with _locked():
        entries = read_store()
        if name not in entries:
            raise ValidationError('Сохранённый ключ с таким именем не найден.')
        del entries[name]
        _write(entries)
