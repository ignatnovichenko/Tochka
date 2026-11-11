"""Check the bundled native Python runtime without downloading dependencies."""
import argparse
import json
from .business.schema import ValidationError


def main(argv=None, *, input_fn=None, print_fn=None):
    print_fn = print_fn or print
    parser = argparse.ArgumentParser(prog='tochka setup', description='Проверить собственный Python-движок Tochka; без запросов к моделям')
    parser.add_argument('--check', action='store_true', help='Проверить готовность локального движка')
    parser.parse_args(argv)
    from .native_backend import runtime_status
    try:
        result = runtime_status()
        print_fn(json.dumps(result, ensure_ascii=False, indent=2))
        if not result.get('ready'):
            print_fn('Движок Tochka не готов. Проверьте установку Python-пакета.')
            return 2
        print_fn('Собственный Python-движок Tochka готов. Дополнительная установка не требуется.')
        return 0
    except (EOFError, KeyboardInterrupt):
        print_fn('Проверка отменена.')
        return 130
    except ValidationError as exc:
        print_fn(str(exc))
        return 2
    except (ValueError, OSError, RuntimeError):
        print_fn('Не удалось проверить движок Tochka. Проверьте установку Python-пакета.')
        return 2
