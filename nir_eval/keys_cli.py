"""User-facing local credential management; no network calls."""
import argparse
import sys
from .business.schema import ValidationError
from .credentials import hidden_key_input, GO_BASE_URL, GO_KEY_ENV
from .key_store import read_store, save_key, delete_key, store_path


def listing():
    entries = read_store()
    print('Локальные ключи Tochka: ' + str(store_path()))
    for name, entry in entries.items():
        print(f"{name} | {entry['base_url']} | {entry['api_key_env']} | ********")
    if not entries:
        print('Сохранённых ключей пока нет.')
    print('Ключи окружения и OpenCode хранятся отдельно; удаление здесь их не изменяет.')


def add(name, base, env, replace=False):
    from .api_support import validate_api_config
    config = {'base_url': base, 'api_key_env': env}
    validate_api_config(config, allow_remote=True)
    key = hidden_key_input('Вставьте API-ключ (ввод скрыт; будет сохранён на этом компьютере): ')
    saved = save_key(config, key, name, replace=replace)
    print('Ключ сохранён: ' + saved)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='tochka keys', description='Локальные ключи: список, добавление, удаление')
    sub = parser.add_subparsers(dest='action')
    sub.add_parser('list', help='Показать список с замаскированными ключами')
    a = sub.add_parser('add', help='Добавить ключ скрытым вводом')
    a.add_argument('name'); a.add_argument('--base-url', default=GO_BASE_URL)
    a.add_argument('--api-key-env', default=GO_KEY_ENV); a.add_argument('--replace', action='store_true')
    d = sub.add_parser('delete', help='Удалить локально сохранённый ключ'); d.add_argument('name')
    s = sub.add_parser('show', help='Явно показать полный ключ в терминале'); s.add_argument('name')
    args = parser.parse_args(argv)
    try:
        if args.action == 'list': listing()
        elif args.action == 'add': add(args.name, args.base_url, args.api_key_env, args.replace)
        elif args.action == 'delete':
            delete_key(args.name); print('Ключ удалён из Tochka. У провайдера он не отозван.')
        elif args.action == 'show':
            if not sys.stdout.isatty():
                raise ValidationError('Полный ключ можно показать только в интерактивном терминале.')
            entry = read_store().get(args.name)
            if entry is None: raise ValidationError('Ключ не найден.')
            print(entry['key'])
        else:
            if not sys.stdin.isatty():
                listing(); return 0
            while True:
                listing()
                action = input('\n1 — добавить, 2 — удалить, Enter — выйти: ').strip()
                if not action: return 0
                if action == '1':
                    name = input('Название подключения: ').strip()
                    base = input('Адрес API [OpenCode Go]: ').strip() or GO_BASE_URL
                    env = input('Имя ключа в настройках [OPENCODE_GO_API_KEY]: ').strip() or GO_KEY_ENV
                    add(name, base, env)
                elif action == '2':
                    name = input('Название удаляемого подключения: ').strip()
                    delete_key(name); print('Ключ удалён из Tochka. У провайдера он не отозван.')
                else: print('Выберите 1, 2 или Enter.')
        return 0
    except (EOFError, KeyboardInterrupt):
        print('\nДействие отменено.'); return 130
    except (ValueError, OSError) as exc:
        print('Ошибка: ' + (str(exc) if isinstance(exc, ValidationError) else 'Не удалось обновить хранилище ключей.'))
        return 2
