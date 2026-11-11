"""Offline doctor privacy, bounded reads and no-execution contract."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from nir_eval.doctor_diagnostics import collect_diagnostics


class DoctorDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {'XDG_CONFIG_HOME': str(self.root / 'config'), 'SECRET_TEST_VALUE': 'ENV_SECRET_NEVER_EXPORT'}).start()

    def collect(self, experiment=None):
        with patch('socket.socket', side_effect=AssertionError('network forbidden')), patch('subprocess.Popen', side_effect=AssertionError('subprocess forbidden')):
            result = collect_diagnostics(experiment)
        self.assertNotIn('ENV_SECRET_NEVER_EXPORT', json.dumps(result))
        self.assertEqual(result['network_calls'], 0)
        self.assertFalse(result['files_modified'])
        return result

    def test_default_checks_are_offline_and_absent_store_is_not_error(self):
        result = self.collect()
        self.assertEqual(result['key_store']['status'], 'absent')
        self.assertTrue(all(x['status'] == 'ok' for x in result['source_syntax']))
        self.assertFalse((self.root / 'config').exists())

    def test_source_syntax_does_not_execute_or_export_code(self):
        source = self.root / 'package'; source.mkdir()
        (source/'app.py').write_text("raise Exception('SOURCE_SECRET_NEVER_EXPORT')\n")
        (source/'broken.py').write_text('TOKEN_SECRET_NEVER_EXPORT = (\n')
        with patch('nir_eval.doctor_diagnostics.PACKAGE_ROOT', source), patch('nir_eval.doctor_diagnostics.SOURCE_FILES', ('app.py','broken.py','absent.py')):
            result = self.collect()
        self.assertEqual([x['status'] for x in result['source_syntax']], ['ok','syntax_error','missing'])
        self.assertNotIn('SECRET_NEVER_EXPORT', json.dumps(result))

    def test_valid_store_has_count_only(self):
        path=self.root/'config/tochka/credentials.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'version':1,'keys':{'private-alias':{'base_url':'https://private.example/v1','api_key_env':'PRIVATE_KEY','key':'CREDENTIAL_SECRET_NEVER_EXPORT'}}}))
        path.chmod(0o600)
        before=path.read_bytes()
        result=self.collect()
        self.assertEqual(result['key_store'], {'status':'ok','stored_entries':1})
        serialized=json.dumps(result)
        for secret in ['private-alias','private.example','PRIVATE_KEY','CREDENTIAL_SECRET_NEVER_EXPORT']:
            self.assertNotIn(secret,serialized)
        self.assertEqual(path.read_bytes(), before)

    def test_unsafe_key_permissions_and_symlink_do_not_repair(self):
        path=self.root/'config/tochka/credentials.json';path.parent.mkdir(parents=True)
        path.write_text('{}');path.chmod(0o644)
        self.assertEqual(self.collect()['key_store']['status'],'unsafe_permissions')
        self.assertEqual(path.stat().st_mode & 0o777,0o644)
        path.unlink(); target=self.root/'secret'; target.write_text('SECRET_NEVER_EXPORT');path.symlink_to(target)
        self.assertEqual(self.collect()['key_store']['status'],'unreadable_or_invalid')
        self.assertTrue(path.is_symlink())

    def test_plan_counts_are_safe_and_credentials_not_checked(self):
        from nir_eval.experiment import main
        root=self.root/'experiment'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['init',str(root),'--demo']),0)
            self.assertEqual(main(['solver-add',str(root),'--provider','api','--base-url','https://private.example/v1','--model','PRIVATE_MODEL','--name','base']),0)
            self.assertEqual(main(['judge-set',str(root),'--provider','api','--base-url','https://private.example/v1','--model','PRIVATE_JUDGE']),0)
        from nir_eval.experiment import experiment_plan
        with patch('nir_eval.experiment.experiment_plan', wraps=experiment_plan) as plan:
            result=self.collect(root)
            plan.assert_called_once_with(root,allow_remote=True,check_credentials=False)
        self.assertEqual(result['experiment']['status'],'valid_plan')
        serialized=json.dumps(result)
        self.assertNotIn('PRIVATE_MODEL',serialized)
        self.assertNotIn('private.example',serialized)
        self.assertNotIn('customer',serialized)

    def test_arbitrary_exception_message_never_exported(self):
        root=self.root/'run';root.mkdir()
        (root/'experiment-manifest.json').write_text('{"secret":"TASK_SECRET_NEVER_EXPORT"')
        result=self.collect(root)
        self.assertEqual(result['experiment']['status'],'invalid_manifest')
        self.assertNotIn('TASK_SECRET_NEVER_EXPORT',json.dumps(result))

    def test_saved_run_summary_does_not_read_answers(self):
        root=self.root/'run';root.mkdir()
        (root/'experiment-manifest.json').write_text(json.dumps({'state':'interrupted','runs':['private-model'],'planned_runs':['private-model','other'],'error':'CREDENTIAL_SECRET_NEVER_EXPORT'}))
        (root/'records.jsonl').write_text('USER_ANSWER_SECRET_NEVER_EXPORT')
        before={p.name:p.read_bytes() for p in root.iterdir()}
        result=self.collect(root)
        self.assertEqual(result['experiment']['runs_count'],1)
        self.assertFalse(result['experiment']['integrity_checked'])
        self.assertEqual(result['findings'][-1]['repair'],'RESUME_TO_NEW_DIRECTORY')
        self.assertNotIn('SECRET_NEVER_EXPORT',json.dumps(result))
        self.assertNotIn('private-model',json.dumps(result))
        self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})

    def test_oversized_manifest_and_path_traversal_refused(self):
        root=self.root/'run';root.mkdir()
        (root/'experiment-manifest.json').write_bytes(b' ' * 1_000_001)
        self.assertEqual(self.collect(root)['experiment']['status'],'invalid_manifest')
        (root/'experiment-manifest.json').unlink()
        (root/'experiment.json').write_text(json.dumps({'solvers':['../secret.json'],'judge':'judge.json'}))
        with patch('nir_eval.experiment.experiment_plan') as plan:
            self.assertEqual(self.collect(root)['experiment']['status'],'invalid_plan')
            plan.assert_not_called()

    def test_missing_completed_artifacts_reported_without_reading_content(self):
        root=self.root/'run';root.mkdir()
        (root/'experiment-manifest.json').write_text(json.dumps({'state':'completed'}))
        result=self.collect(root)
        self.assertEqual(result['experiment']['missing_artifacts'],['comparison.json','report.html'])
        self.assertEqual(result['findings'][-1]['code'],'RUN_ARTIFACTS_MISSING')


if __name__ == '__main__':
    unittest.main()
