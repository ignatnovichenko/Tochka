"""Installer shell contract using isolated fake tools; no network or real installs."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from nir_eval.node_runtime import NODE_VERSION, NODE_SHA256

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
                        TEST_OS='Linux',TEST_ARCH='x86_64',TEST_UV_STATUS='0',TEST_HELP_STATUS='0',TEST_SETUP_STATUS='0',TEST_NODE_STATUS='0',TEST_CURL_STATUS='37',TEST_NODE_SHA=NODE_SHA256['linux-x64'],TEST_DOWNLOAD_URL=str(self.root/'download-url'),TEST_TAR_STATUS='0',TEST_EXTRACTED_NODE_STATUS='0')
        for name in ['sh','mktemp','rm','grep','sed','mkdir','mv','chmod','cat']:
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
        self.bootstrap_fixtures()


    def bootstrap_fixtures(self):
        self.script(self.commands/'curl', """printf 'CURL\n' >> "$TEST_LOG"
[ "$TEST_CURL_STATUS" = 0 ] || exit "$TEST_CURL_STATUS"
while [ "$#" -gt 0 ]; do
  case "$1" in https:*) printf '%s\n' "$1" > "$TEST_DOWNLOAD_URL";; -o) shift; printf 'fixture archive' > "$1";; esac
  shift
done
""")
        self.script(self.commands/'sha256sum', 'printf \'%s  %s\\n\' "$TEST_NODE_SHA" "$1"\n')
        self.script(self.commands/'tar', """printf 'TAR\n' >> "$TEST_LOG"
[ "$TEST_TAR_STATUS" = 0 ] || exit "$TEST_TAR_STATUS"
while [ "$#" -gt 0 ]; do
  if [ "$1" = '-C' ]; then shift; destination=$1; fi
  shift
done
case "$TEST_OS-$TEST_ARCH" in
  Linux-x86_64) platform=linux-x64;; Linux-aarch64|Linux-arm64) platform=linux-arm64;;
  Darwin-x86_64) platform=darwin-x64;; Darwin-arm64|Darwin-aarch64) platform=darwin-arm64;;
esac
mkdir -p "$destination/node-v24.22.2-$platform/bin"
cat > "$destination/node-v24.22.2-$platform/bin/node" <<'NODE'
#!/bin/sh
printf 'MANAGED_NODE\n' >> "$TEST_LOG"
[ "$TEST_EXTRACTED_NODE_STATUS" = 0 ] || exit "$TEST_EXTRACTED_NODE_STATUS"
case "$1" in --version) printf 'v24.22.2\n';; *) exit 0;; esac
NODE
cat > "$destination/node-v24.22.2-$platform/bin/npm" <<'NPM'
#!/bin/sh
exit 0
NPM
chmod +x "$destination/node-v24.22.2-$platform/bin/node" "$destination/node-v24.22.2-$platform/bin/npm"
""")

    def script(self,path,body):
        path.write_text('#!/bin/sh\n'+body);path.chmod(0o755)

    def run_installer(self,*args,**env):
        return subprocess.run(['/bin/sh',str(INSTALLER),*args],env={**self.env,**env},capture_output=True,text=True,timeout=10)

    def calls(self):
        return self.log.read_text() if self.log.exists() else ''

    def test_linux_install_source_version_and_space_paths(self):
        result=self.run_installer('--version','v0.5.0')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('UV<tool><install><--python><3.11><--force><https://github.com/ignatnovichenko/Tochka/archive/refs/tags/v0.5.0.tar.gz>',self.calls())
        self.assertIn('TOCHKA_HELP\nTOCHKA_SETUP',self.calls())
        self.assertIn('UV<tool><update-shell>',self.calls())
        self.assertIn(str(self.bin/'tochka'),result.stdout)
        self.assertNotIn('CURL',self.calls())

    def test_default_release_and_required_runtime_setup(self):
        result=self.run_installer('--no-modify-path')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('/v0.5.0.tar.gz>',self.calls())
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

    def test_missing_node_or_npm_bootstraps_locally(self):
        for name in ['node','npm']:
            with self.subTest(name=name):
                tool=self.commands/name
                body=tool.read_text(); tool.unlink()
                result=self.run_installer(TEST_CURL_STATUS='0')
                tool.write_text(body);tool.chmod(0o755)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('TOCHKA_SETUP',self.calls())
                target=self.root/'data files/tochka/node'/f'node-v{NODE_VERSION}-linux-x64'
                self.assertTrue((target/'bin/node').exists())
                self.assertTrue((target/'bin/npm').exists())
                self.assertFalse(list(target.parent.glob('.node-install.*')))

    def test_old_node_uses_managed_runtime_without_replacing_system(self):
        original=(self.commands/'node').read_bytes()
        result=self.run_installer(TEST_NODE_STATUS='1',TEST_CURL_STATUS='0')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('MANAGED_NODE',self.calls())
        self.assertEqual((self.commands/'node').read_bytes(),original)
        self.assertEqual((self.root/'download-url').read_text().strip(),f'https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-linux-x64.tar.gz')

    def test_managed_node_platforms_have_pinned_official_checksums(self):
        for system,arch,target in [('Linux','aarch64','linux-arm64'),('Darwin','x86_64','darwin-x64'),('Darwin','arm64','darwin-arm64')]:
            with self.subTest(target=target):
                result=self.run_installer(TEST_NODE_STATUS='1',TEST_CURL_STATUS='0',TEST_OS=system,TEST_ARCH=arch,TEST_NODE_SHA=NODE_SHA256[target])
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertTrue((self.root/'data files/tochka/node'/f'node-v{NODE_VERSION}-{target}'/'bin/npm').is_file())

    def test_checksum_failure_never_extracts_or_installs(self):
        result=self.run_installer(TEST_NODE_STATUS='1',TEST_CURL_STATUS='0',TEST_NODE_SHA='incorrect')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('checksum mismatch',result.stderr)
        self.assertNotIn('TAR',self.calls())
        self.assertNotIn('UV',self.calls())
        parent=self.root/'data files/tochka/node'
        self.assertFalse(list(parent.glob('node-v*')))
        self.assertFalse(list(parent.glob('.node-install.*')))

    def test_download_extract_and_execution_failure_clean_staging(self):
        for env in [{'TEST_CURL_STATUS':'37'}, {'TEST_CURL_STATUS':'0','TEST_TAR_STATUS':'2'},
                    {'TEST_CURL_STATUS':'0','TEST_EXTRACTED_NODE_STATUS':'126'}]:
            with self.subTest(env=env):
                result=self.run_installer(TEST_NODE_STATUS='1',**env)
                self.assertNotEqual(result.returncode,0)
                parent=self.root/'data files/tochka/node'
                self.assertFalse(list(parent.glob('node-v*')))
                self.assertFalse(list(parent.glob('.node-install.*')))
                self.assertNotIn('TOCHKA_SETUP',self.calls())

    def test_existing_damaged_managed_directory_is_preserved(self):
        target=self.root/'data files/tochka/node'/f'node-v{NODE_VERSION}-linux-x64'
        target.mkdir(parents=True)
        (target/'sentinel').write_text('preserve')
        result=self.run_installer(TEST_NODE_STATUS='1',TEST_CURL_STATUS='0')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('incomplete',result.stderr)
        self.assertEqual((target/'sentinel').read_text(),'preserve')
        self.assertEqual(self.calls(),'')

    def test_shasum_supported_for_macos(self):
        (self.commands/'sha256sum').unlink()
        self.script(self.commands/'shasum', 'printf \'%s  %s\\n\' "$TEST_NODE_SHA" "$3"\n')
        result=self.run_installer(TEST_NODE_STATUS='1',TEST_CURL_STATUS='0',TEST_OS='Darwin',TEST_ARCH='arm64',TEST_NODE_SHA=NODE_SHA256['darwin-arm64'])
        self.assertEqual(result.returncode,0,result.stderr)

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
