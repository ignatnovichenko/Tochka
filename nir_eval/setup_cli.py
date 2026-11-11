"""Install or check the required Promptfoo runtime without model inference."""
import argparse
import json
from .business.schema import ValidationError


def main(argv=None, *, input_fn=None, print_fn=None):
    print_fn = print_fn or print
    parser = argparse.ArgumentParser(prog='tochka setup', description='Установить Promptfoo; модельные запросы не выполняются')
    parser.add_argument('--check', action='store_true', help='Только проверить готовность, без установки')
    args = parser.parse_args(argv)
    from .promptfoo_backend import runtime_status, setup_runtime
    try:
        if args.check:
            result = runtime_status()
        else:
            print_fn('Установка закреплённой версии Promptfoo. Модельные запросы не выполняются.')
            result = setup_runtime()
        print_fn(json.dumps(result, ensure_ascii=False, indent=2))
        if not result.get('ready'):
            print_fn('Promptfoo не готов. Проверьте Node.js/npm и выполните tochka setup.')
            return 2
        print_fn('Promptfoo готов.')
        return 0
    except (EOFError, KeyboardInterrupt):
        print_fn('Установка отменена.'); return 130
    except ValidationError as exc:
        print_fn(str(exc)); return 2
    except (ValueError, OSError, RuntimeError):
        print_fn('Не удалось установить Promptfoo. Проверьте Node.js/npm и доступ к сети; повторите tochka setup.'); return 2
