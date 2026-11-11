"""Runtime installation and lifecycle tests; no npm, Node or network execution."""
import gc
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import weakref
from unittest.mock import Mock, patch

from nir_eval import promptfoo_backend as backend
from nir_eval.business.schema import ValidationError


class PromptfooRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = patch.dict(os.environ, {'XDG_CONFIG_HOME': str(self.root/'config'),
                                     'XDG_DATA_HOME': str(self.root/'data')})
        env.start(); self.addCleanup(env.stop)
        self.status = {'ready': False, 'node_compatible': True}

    def test_setup_copies_pinned_lock_and_disables_install_scripts(self):
        target = backend.install_path()
        def install(args, **kwargs):
            self.assertEqual(args, ['/fixture/npm', 'ci', '--ignore-scripts', '--no-audit', '--no-fund'])
            stage = kwargs['cwd']
            for name in ('package.json', 'package-lock.json', 'worker.mjs'):
                self.assertEqual((stage/name).read_bytes(), (backend.BUNDLED/name).read_bytes())
            manifest = json.loads((stage/'package.json').read_text())
            lock = json.loads((stage/'package-lock.json').read_text())
            self.assertEqual(manifest['dependencies']['promptfoo'], backend.VERSION)
            self.assertEqual(lock['packages']['node_modules/promptfoo']['version'], backend.VERSION)
            self.assertEqual(kwargs['stdout'], subprocess.DEVNULL)
            self.assertEqual(kwargs['stderr'], subprocess.DEVNULL)
            self.assertEqual(kwargs['timeout'], 900)
            return Mock(returncode=0)
        with patch.object(backend, 'runtime_status', side_effect=[self.status, self.status, {'ready': True}]), \
             patch.object(backend.shutil, 'which', return_value='/fixture/npm'), \
             patch.object(backend.subprocess, 'run', side_effect=install) as run:
            self.assertTrue(backend.setup_runtime()['ready'])
        run.assert_called_once()
        self.assertTrue((target/'package-lock.json').is_file())
        self.assertFalse(list(target.parent.glob('.promptfoo-*')))

    def test_failed_setup_cleans_stage(self):
        for error in (Mock(returncode=1), subprocess.TimeoutExpired('npm', 900)):
            with self.subTest(error=type(error).__name__), \
                 patch.object(backend, 'runtime_status', return_value=self.status), \
                 patch.object(backend.shutil, 'which', return_value='/fixture/npm'), \
                 patch.object(backend.subprocess, 'run') as run:
                if isinstance(error, Exception): run.side_effect = error
                else: run.return_value = error
                with self.assertRaises(ValidationError): backend.setup_runtime()
            self.assertFalse(backend.install_path().exists())
            self.assertFalse(list(backend.install_path().parent.glob('.promptfoo-*')))

    def test_setup_rejects_failed_final_health_check(self):
        with patch.object(backend, 'runtime_status', return_value=self.status), \
             patch.object(backend.shutil, 'which', return_value='/fixture/npm'), \
             patch.object(backend.subprocess, 'run', return_value=Mock(returncode=0)):
            with self.assertRaisesRegex(ValidationError, 'Promptfoo|runtime|готов|setup'):
                backend.setup_runtime()
        self.assertFalse(list(backend.install_path().parent.glob('.promptfoo-*')))

    def test_invalid_explicit_override_does_not_fall_back(self):
        override = self.root/'nonexistent-runtime'
        with patch.dict(os.environ, {'TOCHKA_PROMPTFOO_DIR': str(override)}), \
             patch.object(backend.shutil, 'which', return_value=None):
            self.assertEqual(backend.runtime_path(), override.resolve())
            self.assertFalse(backend.runtime_status()['ready'])
            with self.assertRaises(ValidationError): backend.require_runtime()

    def test_health_check_failure_is_not_ready_and_closes_worker(self):
        runtime = self.root/'runtime'
        package = runtime/'node_modules/promptfoo/package.json'
        package.parent.mkdir(parents=True)
        package.write_text(json.dumps({'version': backend.VERSION}))
        (runtime/'worker.mjs').write_text('// fixture')
        with patch.object(backend, 'runtime_path', return_value=runtime), \
             patch.object(backend, '_VERIFIED', set()), \
             patch.object(backend.shutil, 'which', return_value='/fixture/node'), \
             patch.object(backend.subprocess, 'run', return_value=Mock(stdout='v24.0.0')), \
             patch.object(backend, 'Worker') as worker:
            worker.return_value.call.return_value = {'ok': False, 'version': backend.VERSION}
            self.assertFalse(backend.runtime_status()['ready'])
            worker.return_value.call.assert_called_once_with({'op':'health', 'timeout_seconds':10})
            worker.return_value.close.assert_called_once_with()

    def test_ready_runtime_does_not_install(self):
        with patch.object(backend, 'runtime_status', return_value={'ready': True}), \
             patch.object(backend.subprocess, 'run') as run:
            self.assertTrue(backend.setup_runtime()['ready'])
        run.assert_not_called()

    def test_worker_collection_closes_process_without_bound_atexit_refs(self):
        process = Mock(pid=123456)
        process.poll.return_value = None
        scratch = Mock()
        with patch.object(backend.atexit, 'register') as register, patch.object(backend.os, 'killpg') as kill:
            worker = backend.Worker(self.root)
            worker.proc = process
            worker.scratch = scratch
            ref = weakref.ref(worker)
            self.assertIn(worker, backend._WORKERS)
            del worker
            gc.collect()
            self.assertIsNone(ref())
            register.assert_not_called()
            kill.assert_called_once_with(123456, backend.signal.SIGKILL)
        process.wait.assert_called_once_with()
        process.stdin.close.assert_called_once_with()
        process.stdout.close.assert_called_once_with()
        scratch.cleanup.assert_called_once_with()

    def test_shutdown_closes_live_workers_idempotently(self):
        worker = backend.Worker(self.root)
        process = Mock(); process.poll.return_value = 0
        worker.proc = process
        with patch.object(backend, '_WORKERS', weakref.WeakSet([worker])):
            backend._close_workers()
            backend._close_workers()
        process.wait.assert_called_once_with()
        self.assertIsNone(worker.proc)


if __name__ == '__main__':
    unittest.main()
