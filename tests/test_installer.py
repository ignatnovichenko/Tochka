"""Installer shell contract using isolated fake tools; no network or real installs."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

INSTALLER = Path(__file__).resolve().parents[1] / 'install.sh'


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='tochka installer ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.commands = self.root / 'fake commands'; self.commands.mkdir()
        self.bin = self.root / 'installed applications'; self.bin.mkdir()
        self.log = self.root / 'calls.log'
        self.env = dict(os.environ)
        for name in ['TOCHKA_INSTALL_SOURCE','UV_INSTALL_DIR','UV_NO_MODIFY_PATH']:
            self.env.pop(name,None)
        self.env.update(PATH=str(self.commands), XDG_DATA_HOME=str(self.root/'data files'),
                        XDG_CONFIG_HOME=str(self.root/'config files'), XDG_STATE_HOME=str(self.root/'state files'),
                        TMPDIR=str(self.root), TEST_LOG=str(self.log), TEST_BIN=str(self.bin),
                        TEST_OS='Linux',TEST_ARCH='x86_64',TEST_UV_STATUS='0',TEST_HELP_STATUS='0',TEST_SETUP_STATUS='0',TEST_NODE_STATUS='0',TEST_CURL_STATUS='37')
        for name in ['sh','mktemp','rm','grep','sed']:
            binary=shutil.which(name)
            if binary is None:self.skipTest(name+' unavailable')
            (self.commands/name).symlink_to(binary)
        self.script(self.commands/'uname', 'case "$1" in -s) printf "%s\\n" "$TEST_OS";; -m) printf "%s\\n" "$TEST_ARCH";; esac\n')
        self.script(self.commands/'uv', '''printf 'UV' >> "$TEST_LOG"
for argument in "$@"; do printf '<%s>' "$argument" >> "$TEST_LOG"; done
printf '\\n' >> "$TEST_LOG"
case "$*" in
  'tool dir --bin') printf '%s\\n' "$TEST_BIN" ;;
  'tool update-shell') exit "${TEST_SHELL_STATUS:-0}" ;;
  *) exit "$TEST_UV_STATUS" ;;
esac
''')
        self.script(self.commands/'node', 'exit "$TEST_NODE_STATUS"\n')
        self.script(self.commands/'npm', 'exit 0\n')
        self.script(self.bin/'tochka', '''case "$1" in
  --help) printf 'TOCHKA_HELP\\n' >> "$TEST_LOG"; exit "$TEST_HELP_STATUS" ;;
  setup) printf 'TOCHKA_SETUP\\n' >> "$TEST_LOG"; exit "$TEST_SETUP_STATUS" ;;
  *) exit 99 ;;
esac
''')
        self.script(self.commands/'curl', 'printf "CURL\\n" >> "$TEST_LOG"\nexit "$TEST_CURL_STATUS"\n')

    def script(self,path,body):
        path.write_text('#!/bin/sh\n'+body);path.chmod(0o755)

    def run_installer(self,*args,**env):
        return subprocess.run(['/bin/sh',str(INSTALLER),*args],env={**self.env,**env},capture_output=True,text=True,timeout=10)

    def calls(self):
        return self.log.read_text() if self.log.exists() else ''

    def test_linux_install_source_version_and_space_paths(self):
        result=self.run_installer('--version','v0.4.0')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('UV<tool><install><--python><3.11><--force><https://github.com/ignatnovichenko/Tochka/archive/refs/tags/v0.4.0.tar.gz>',self.calls())
        self.assertIn('TOCHKA_HELP\nTOCHKA_SETUP',self.calls())
        self.assertIn('UV<tool><update-shell>',self.calls())
        self.assertIn(str(self.bin/'tochka'),result.stdout)
        self.assertNotIn('CURL',self.calls())

    def test_default_release_and_required_runtime_setup(self):
        result=self.run_installer('--no-modify-path')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('/v0.4.0.tar.gz>',self.calls())
        self.assertIn('TOCHKA_HELP\nTOCHKA_SETUP',self.calls())
        self.assertIn('Promptfoo runtime is ready',result.stdout)

    def test_setup_failure_reports_partial_install_without_success(self):
        result=self.run_installer(TEST_SETUP_STATUS='2')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('TOCHKA_SETUP',self.calls())
        self.assertNotIn('Tochka installed.',result.stdout)
        self.assertNotIn('runtime is ready',result.stdout)
        self.assertNotIn('update-shell',self.calls())
        self.assertIn('CLI was installed, but Promptfoo setup failed',result.stderr)
        self.assertIn(str(self.bin/'tochka'),result.stderr)

    def test_missing_node_or_npm_rejected_before_install(self):
        for name in ['node','npm']:
            with self.subTest(name=name):
                tool=self.commands/name
                body=tool.read_text()
                tool.unlink()
                result=self.run_installer()
                tool.write_text(body);tool.chmod(0o755)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('Node.js >=22.22.0 and npm',result.stderr)
                self.assertEqual(self.calls(),'')

    def test_old_node_rejected_before_install(self):
        result=self.run_installer(TEST_NODE_STATUS='1')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Node.js >=22.22.0 is required',result.stderr)
        self.assertEqual(self.calls(),'')

    def test_darwin_arm_install_no_modify_path(self):
        result=self.run_installer('--version','1.2.3','--no-modify-path',TEST_OS='Darwin',TEST_ARCH='arm64')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('/v1.2.3.tar.gz>',self.calls())
        self.assertNotIn('update-shell',self.calls())
        self.assertIn('TOCHKA_HELP\nTOCHKA_SETUP',self.calls())

    def test_source_override_is_single_argument_even_with_spaces(self):
        source=str(self.root/'local source release.whl')
        result=self.run_installer('--no-modify-path',TOCHKA_INSTALL_SOURCE=source)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('<--force><'+source+'>\n',self.calls())

    def test_uv_failure_cannot_report_success(self):
        result=self.run_installer(TEST_UV_STATUS='42')
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn('Tochka installed.',result.stdout)
        self.assertNotIn('TOCHKA_HELP',self.calls())
        self.assertNotIn('update-shell',self.calls())

    def test_installed_command_failure_cannot_report_success(self):
        result=self.run_installer(TEST_HELP_STATUS='17')
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn('Tochka installed.',result.stdout)
        self.assertNotIn('update-shell',self.calls())

    def test_unsupported_os_and_arch_rejected_before_tools(self):
        for env in [{'TEST_OS':'FreeBSD'},{'TEST_ARCH':'i686'}]:
            with self.subTest(env=env):
                result=self.run_installer(**env)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(self.calls(),'')

    def test_invalid_versions_rejected_without_installer_calls(self):
        for version in ['', '1.2.3;touch /tmp/no', '../main','latest','.', '1..2','.3','3.']:
            with self.subTest(version=version):
                if self.log.exists():self.log.unlink()
                result=self.run_installer('--version',version)
                self.assertNotEqual(result.returncode,0,result.stdout)
                self.assertEqual(self.calls(),'')

    def test_missing_version_and_unknown_option(self):
        for args in [('--version',),('--unknown',)]:
            with self.subTest(args=args):
                result=self.run_installer(*args)
                self.assertEqual(result.returncode,2)
                self.assertEqual(self.calls(),'')

    def test_bootstrap_download_failure_stops_without_install_and_cleans_temp(self):
        (self.commands/'uv').unlink()
        result=self.run_installer()
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(self.calls(),'CURL\n')
        self.assertNotIn('Tochka installed.',result.stdout)
        self.assertFalse((self.root/'data files/tochka/bootstrap/uv').exists())
        self.assertFalse(list(self.root.glob('tochka-install.*')))

    def test_path_update_failure_reports_absolute_launch_command(self):
        result=self.run_installer(TEST_SHELL_STATUS='3')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('PATH could not be updated.',result.stderr)
        self.assertIn(str(self.bin/'tochka'),result.stderr)
        self.assertIn('Tochka installed.',result.stdout)


if __name__=='__main__':unittest.main()
