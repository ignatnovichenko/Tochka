"""Updater contract without changing the installed application or using network."""
import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from urllib.error import URLError
from unittest.mock import MagicMock, patch

from nir_eval import app, update_cli
from nir_eval.interactive import MENU_COMMANDS, run_shell


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.response = MagicMock()
        self.response.__enter__.return_value = self.response
        self.response.geturl.return_value = update_cli.INSTALLER_URL
        self.response.read.return_value = b"#!/bin/sh\n    tochka_version='99.0.0'\nexit 0\n"

    def test_older_published_release_does_not_replace_local_changes(self):
        self.response.read.return_value = b"#!/bin/sh\n    tochka_version='0.5.1'\n"
        with patch.object(update_cli, '__version__', '0.5.2.dev0'), \
             patch.object(update_cli, 'urlopen', return_value=self.response), \
             patch.object(update_cli.subprocess, 'run') as install:
            self.assertEqual(update_cli.main([], print_fn=self.lines.append), 0)
            install.assert_not_called()
        self.assertIn('новее', '\n'.join(self.lines))

    def test_official_installer_runs_without_dev_source_or_shell_profile_change(self):
        def run(args, *, env):
            self.assertEqual(args[0], 'sh')
            self.assertEqual(args[2:], ['--no-modify-path'])
            self.assertEqual(Path(args[1]).read_bytes(), self.response.read.return_value)
            self.assertNotIn('TOCHKA_INSTALL_SOURCE', env)
            return SimpleNamespace(returncode=0)
        with patch.dict(os.environ, {'TOCHKA_INSTALL_SOURCE': '/unpublished/code'}), \
             patch.object(update_cli, 'urlopen', return_value=self.response) as download, \
             patch.object(update_cli.subprocess, 'run', side_effect=run) as install:
            self.assertEqual(update_cli.main([], print_fn=self.lines.append), 0)
        download.assert_called_once_with(update_cli.INSTALLER_URL, timeout=10)
        install.assert_called_once()
        self.assertIn('заново', '\n'.join(self.lines))

    def test_invalid_or_oversized_download_never_executes(self):
        for data in [b'<html>error</html>', b'#!/bin/sh\n' + b'x' * update_cli.MAX_INSTALLER_BYTES]:
            self.response.read.return_value = data
            with patch.object(update_cli, 'urlopen', return_value=self.response), \
                 patch.object(update_cli.subprocess, 'run') as install:
                self.assertEqual(update_cli.main([], print_fn=self.lines.append), 2)
                install.assert_not_called()

    def test_primary_timeout_uses_official_api_and_keeps_checks(self):
        with patch.object(update_cli, 'urlopen', side_effect=[URLError(TimeoutError()), self.response]) as download, \
             patch.object(update_cli.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
            self.assertEqual(update_cli.main([], print_fn=self.lines.append), 0)
        request = download.call_args_list[1].args[0]
        self.assertEqual(request.full_url, update_cli.INSTALLER_API_URL)
        self.assertEqual(request.get_header('Accept'), 'application/vnd.github.raw+json')
        self.assertIn('истекло время', '\n'.join(self.lines))

    def test_both_sources_fail_with_specific_reason(self):
        with patch.object(update_cli, 'urlopen', side_effect=URLError(TimeoutError())), \
             patch.object(update_cli.subprocess, 'run') as install:
            self.assertEqual(update_cli.main([], print_fn=self.lines.append), 2)
        install.assert_not_called()
        self.assertIn('api.github.com: истекло время', '\n'.join(self.lines))

    def test_failure_does_not_report_success(self):
        for error in [OSError('fixture'), KeyboardInterrupt()]:
            with patch.object(update_cli, 'urlopen', side_effect=error), \
                 patch.object(update_cli.subprocess, 'run') as install:
                self.assertNotEqual(update_cli.main([], print_fn=self.lines.append), 0)
                install.assert_not_called()
        with patch.object(update_cli, 'urlopen', return_value=self.response), \
             patch.object(update_cli.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
            self.assertEqual(update_cli.main([], print_fn=self.lines.append), 2)
        self.assertNotIn('Обновление завершено.', '\n'.join(self.lines))

    def test_help_has_no_side_effects(self):
        with patch.object(update_cli, 'urlopen') as download, contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaises(SystemExit) as exc:
            update_cli.main(['--help'])
        self.assertEqual(exc.exception.code, 0)
        download.assert_not_called()

    def test_terminal_dispatch(self):
        with patch.object(update_cli, 'main', return_value=0) as update:
            self.assertEqual(app.main(['update']), 0)
        update.assert_called_once_with([])

    def test_slash_and_menu_update_exit_console_on_success(self):
        number = MENU_COMMANDS.index('update') + 1
        for entries in [['/update'], ['/', str(number)]]:
            answers = iter(entries)
            with patch.object(update_cli, 'main', return_value=0) as update:
                self.assertEqual(run_shell(input_fn=lambda _: next(answers), print_fn=self.lines.append), 0)
            update.assert_called_once_with([], print_fn=self.lines.append)

    def test_failed_update_leaves_console_usable(self):
        answers = iter(['/update', '/exit'])
        with patch.object(update_cli, 'main', return_value=2):
            self.assertEqual(run_shell(input_fn=lambda _: next(answers), print_fn=self.lines.append), 0)


if __name__ == '__main__':
    unittest.main()
