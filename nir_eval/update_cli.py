"""Update the installed application using Tochka's published installer."""
import argparse
import os
import re
from pathlib import Path
import subprocess
import tempfile
import socket
import ssl
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from . import __version__

INSTALLER_URL = 'https://raw.githubusercontent.com/ignatnovichenko/Tochka/main/install.sh'
INSTALLER_API_URL = 'https://api.github.com/repos/ignatnovichenko/Tochka/contents/install.sh?ref=main'
MAX_INSTALLER_BYTES = 256 * 1024


class DownloadError(Exception):
    pass


def network_reason(error):
    if isinstance(error, HTTPError):
        return f'HTTP {error.code}'
    reason = error.reason if isinstance(error, URLError) else error
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return 'истекло время ожидания соединения'
    if isinstance(reason, ssl.SSLCertVerificationError):
        return 'не удалось проверить сертификат HTTPS'
    if isinstance(reason, socket.gaierror):
        return 'не удалось определить адрес сервера (DNS)'
    return f'ошибка соединения ({type(reason).__name__})'


def download_installer(print_fn):
    sources = [INSTALLER_URL, Request(INSTALLER_API_URL, headers={
        'Accept': 'application/vnd.github.raw+json', 'User-Agent': 'Tochka-updater'})]
    failures = []
    for index, source in enumerate(sources):
        host = 'raw.githubusercontent.com' if index == 0 else 'api.github.com'
        try:
            with urlopen(source, timeout=10) as response:
                if not response.geturl().startswith('https://'):
                    raise ValueError('HTTPS required')
                data = response.read(MAX_INSTALLER_BYTES + 1)
            if not data.startswith(b'#!/bin/sh\n') or len(data) > MAX_INSTALLER_BYTES:
                raise ValueError('Invalid installer')
            return data
        except (OSError, HTTPException) as exc:
            reason = network_reason(exc)
        except ValueError:
            reason = 'получен некорректный установщик'
        failures.append(f'{host}: {reason}')
        if index == 0:
            print_fn(f'{failures[-1]}. Пробую официальный API GitHub…')
    raise DownloadError('; '.join(failures))


def main(argv=None, *, print_fn=None):
    print_fn = print_fn or print
    parser = argparse.ArgumentParser(prog='tochka update', description='Обновить Tochka из официального GitHub')
    parser.parse_args(argv)
    print_fn('Загружаю установщик обновления с GitHub…')
    try:
        data = download_installer(print_fn)
        remote = re.search(rb"^\s+tochka_version='(\d+)\.(\d+)\.(\d+)'$", data, re.MULTILINE)
        local = re.match(r'^(\d+)\.(\d+)\.(\d+)', __version__)
        if remote is None or local is None:
            raise ValueError('Unknown installer version')
        if tuple(map(int, remote.groups())) < tuple(map(int, local.groups())):
            print_fn('Локальная версия новее опубликованной на GitHub. Обновление не требуется.')
            return 0
        with tempfile.TemporaryDirectory(prefix='tochka-update-') as directory:
            installer = Path(directory) / 'install.sh'
            installer.write_bytes(data)
            env = dict(os.environ)
            # Development overrides must not replace the official update source.
            env.pop('TOCHKA_INSTALL_SOURCE', None)
            result = subprocess.run(['sh', str(installer), '--no-modify-path'], env=env)
        if result.returncode:
            print_fn('Обновление не завершено. Проверьте сообщения установщика и повторите /update.')
            return 2
    except KeyboardInterrupt:
        print_fn('Обновление прервано. Повторите /update для завершения.')
        return 130
    except DownloadError as exc:
        print_fn(f'Не удалось скачать обновление. {exc}. Повторите /update позже.')
        return 2
    except ValueError:
        print_fn('Не удалось распознать версию официального установщика. Обновление остановлено.')
        return 2
    except (OSError, subprocess.SubprocessError):
        print_fn('Не удалось запустить установщик обновления. Проверьте наличие sh и права на установку.')
        return 2
    print_fn('Обновление завершено. Запустите tochka заново, чтобы использовать новую версию.')
    return 0
