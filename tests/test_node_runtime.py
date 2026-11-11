"""Node discovery does not install software or change system configuration."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from nir_eval import node_runtime as node


class NodeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        env=patch.dict(os.environ,{'XDG_DATA_HOME':str(self.root/'data with spaces')})
        env.start(); self.addCleanup(env.stop)

    def test_supported_system_pair_is_preferred(self):
        with patch.object(node.shutil,'which',side_effect=lambda name:'/system/bin/'+name), \
             patch.object(node,'_version',return_value=(24,1,0)):
            self.assertEqual(node.node_command(),'/system/bin/node')
            self.assertEqual(node.npm_command(),'/system/bin/npm')

    def test_compatible_node_without_npm_is_not_usable_pair(self):
        with patch.object(node.shutil,'which',side_effect=lambda name:'/system/bin/node' if name=='node' else None), \
             patch.object(node,'_version',return_value=(24,1,0)):
            self.assertIsNone(node.node_command())
            self.assertIsNone(node.npm_command())

    def test_managed_install_is_found_without_path_changes(self):
        with patch.object(node.platform,'system',return_value='Linux'),patch.object(node.platform,'machine',return_value='x86_64'):
            root=node.managed_root(); (root/'bin').mkdir(parents=True)
            for name in ('node','npm'):
                (root/'bin'/name).write_text('fixture'); (root/'bin'/name).chmod(0o700)
            with patch.object(node.shutil,'which',return_value=None),patch.object(node,'_version',return_value=(24,22,2)):
                self.assertEqual(node.node_command(),str(root/'bin/node'))
                self.assertEqual(node.npm_command(),str(root/'bin/npm'))
                env=node.command_environment({'PATH':'/usr/bin','SAFE':'yes'})
                self.assertEqual(env,{'PATH':str(root/'bin')+os.pathsep+'/usr/bin','SAFE':'yes'})
            with patch.object(node.shutil,'which',return_value=None),patch.object(node,'_version',return_value=(24,22,1)):
                self.assertIsNone(node.node_command())

    def test_platform_mapping_and_unsupported(self):
        for system,machine,expected in [('Linux','x86_64','linux-x64'),('Linux','aarch64','linux-arm64'),
                                        ('Darwin','x86_64','darwin-x64'),('Darwin','arm64','darwin-arm64'),
                                        ('Windows','AMD64',None),('Linux','i686',None)]:
            with self.subTest(system=system,machine=machine),patch.object(node.platform,'system',return_value=system),patch.object(node.platform,'machine',return_value=machine):
                self.assertEqual(node.platform_id(),expected)

    def test_version_probe_is_bounded_and_strict(self):
        for text,expected in [('v22.22.0\n',(22,22,0)),('24.22.2',(24,22,2)),('v24.1.0-nightly',None),('garbage',None)]:
            with patch.object(node.subprocess,'run',return_value=Mock(stdout=text)) as run:
                self.assertEqual(node._version('/fixture/node'),expected)
                self.assertEqual(run.call_args.kwargs['timeout'],5)
        with patch.object(node.subprocess,'run',side_effect=subprocess.TimeoutExpired('node',5)):
            self.assertIsNone(node._version('/fixture/node'))

    def test_environment_does_not_inherit_secrets(self):
        with patch.dict(os.environ,{'SECRET_TOKEN':'private'}),patch.object(node,'node_command',return_value='/private/node/bin/node'):
            env=node.command_environment({'PATH':'/usr/bin'})
            self.assertNotIn('SECRET_TOKEN',env)
            self.assertEqual(env['PATH'],'/private/node/bin:/usr/bin')

    def test_shell_and_python_pins_match(self):
        installer=(Path(__file__).resolve().parents[1]/'install.sh').read_text()
        self.assertIn("node_version='"+node.NODE_VERSION+"'",installer)
        for checksum in node.NODE_SHA256.values(): self.assertIn(checksum,installer)


if __name__=='__main__': unittest.main()
