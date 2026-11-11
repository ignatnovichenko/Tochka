"""Current assistant model; credentials live in the separate key store."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import stat
from .api_support import validate_api_config
from .business.schema import ValidationError
from .credentials import GO_BASE_URL, GO_KEY_ENV


def settings_path():
    return Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config') / 'tochka' / 'assistant.json'


def validate_profile(profile):
    if not isinstance(profile, dict) or set(profile) != {'model', 'provider'}:
        raise ValidationError('Настройка помощника должна содержать модель и API-подключение.')
    model = profile['model']
    if not isinstance(model, str) or not model or len(model) > 200 or any(c.isspace() or ord(c) < 32 for c in model):
        raise ValidationError('Укажите ID модели без пробелов.')
    if model.startswith(('sk-', 'ghp_', 'gho_')):
        raise ValidationError('Нужен ID модели, а не API-ключ.')
    provider = profile['provider']
    if not isinstance(provider, dict) or provider.get('type') != 'openai-compatible':
        raise ValidationError('Помощник поддерживает OpenAI-compatible API.')
    validate_api_config(provider, allow_remote=True, require_key=False)
    return profile


def load_settings():
    p = settings_path()
    try:
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 16000:
                raise ValueError()
            raw = stream.read(16001)
            if len(raw) > 16000: raise ValueError()
        return validate_profile(json.loads(raw))
    except FileNotFoundError: return None
    except (ValueError, TypeError, OSError):
        raise ValidationError('Настройки модели помощника повреждены. Выполните /model для повторной настройки.') from None


def save_settings(profile):
    validate_profile(profile)
    p = settings_path()
    p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if p.parent.is_symlink() or p.is_symlink():
        raise ValidationError('Путь настроек модели не должен быть символической ссылкой.')
    fd, name = tempfile.mkstemp(prefix='.assistant-', dir=p.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(profile, stream, ensure_ascii=False, indent=2)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, p)
    finally:
        if os.path.exists(name): os.unlink(name)


def main(argv=None, *, input_fn=input, print_fn=print):
    p = argparse.ArgumentParser(prog='tochka model', description='Модель для /doctor; без сетевых запросов')
    p.add_argument('--model'); p.add_argument('--base-url'); p.add_argument('--api-key-env')
    args = p.parse_args(argv)
    try:
        try: current = load_settings()
        except ValidationError: current = None
        if args.model:
            profile = {'model': args.model, 'provider': {'type': 'openai-compatible',
                'base_url': args.base_url or (current or {}).get('provider', {}).get('base_url', GO_BASE_URL),
                'api_key_env': args.api_key_env or (current or {}).get('provider', {}).get('api_key_env', GO_KEY_ENV),
                'timeout_seconds': 180}}
        else:
            default_provider = current['provider'] if current else {'base_url': GO_BASE_URL, 'api_key_env': GO_KEY_ENV}
            if current: print_fn('Текущая модель: ' + current['model'])
            model = input_fn('ID модели для помощника' + (' [' + current['model'] + ']' if current else '') + ': ').strip() or (current['model'] if current else '')
            base = input_fn('Адрес API [' + default_provider['base_url'] + ']: ').strip() or default_provider['base_url']
            env = input_fn('Имя ключа [' + default_provider['api_key_env'] + ']: ').strip() or default_provider['api_key_env']
            profile = {'model': model, 'provider': {'type': 'openai-compatible', 'base_url': base, 'api_key_env': env, 'timeout_seconds': 180}}
        save_settings(profile)
        print_fn('Модель помощника сохранена: ' + profile['model'] + '. Запросы не выполнялись.')
        return 0
    except (EOFError, KeyboardInterrupt):
        print_fn('Настройка отменена.'); return 130
    except (ValueError, OSError) as exc:
        print_fn(str(exc) if isinstance(exc, ValidationError) else 'Не удалось сохранить настройки модели.'); return 2
