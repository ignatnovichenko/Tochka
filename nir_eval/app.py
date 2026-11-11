"""Tochka's product CLI; legacy research commands remain under nir-eval."""
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args in [['--help'], ['-h']]:
        from .experiment import parser
        parser().print_help()
        print('\nИнтерактивно: tochka → / → номер действия. Добавьте --help к команде для подробностей.')
        print('Дополнительно: tochka update, tochka doctor, tochka model, tochka shell; добавьте --help для подробностей.')
        return 0
    if not args:
        if sys.stdin.isatty():
            from .interactive import run_shell
            return run_shell()
        from .experiment import parser
        parser().print_help()
        print('Интерактивная консоль: tochka (в терминале). Дополнительно: update, doctor, model, shell.')
        return 0
    if args[0] in {'update', 'wizard', 'doctor', 'model', 'shell'}:
        if args[0] == 'update':
            from .update_cli import main as command
        elif args[0] == 'wizard':
            from .wizard import main as command
        elif args[0] == 'doctor':
            from .doctor import main as command
        elif args[0] == 'model':
            from .doctor_settings import main as command
        else:
            from .interactive import main as command
        return command(args[1:])
    from .experiment import main as experiment
    if args[0] == 'experiment':
        args = args[1:]
    return experiment(args)


if __name__ == '__main__':
    raise SystemExit(main())
