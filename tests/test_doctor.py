"""Model-assisted doctor: mocked generation, no model or network requests."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from nir_eval import doctor
from nir_eval.business.schema import ValidationError
from nir_eval.providers import Generation


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'project';self.source.mkdir()
        (self.source/'pyproject.toml').write_text('[project]\nname="fixture"\n')
        (self.source/'nir_eval').mkdir();self.path=self.source/'nir_eval/app.py'
        self.original='def main():\n    return 1\n';self.path.write_text(self.original)
        self.profile={'model':'fixture-doctor','provider':{'type':'openai-compatible','base_url':'https://api.example.test/v1','api_key_env':'DOCTOR_TEST_KEY'}}
        self.secret='test-private-key-123456789'
        self.addCleanup(patch.stopall)
        patch.dict(os.environ,{'XDG_CONFIG_HOME':str(self.root/'config'),'XDG_STATE_HOME':str(self.root/'state'),'DOCTOR_TEST_KEY':self.secret}).start()
        patch('nir_eval.doctor.saved_go_key',return_value='').start()
        patch('nir_eval.doctor_diagnostics.collect_diagnostics',return_value={'findings':[],'network_calls':0}).start()
        patch('socket.socket',side_effect=AssertionError('network forbidden')).start()
        self.logs=[]

    def advice(self,edits=None,diagnosis='Проверка',steps=None):
        return json.dumps({'diagnosis':diagnosis,'steps':steps or [],'edits':edits or []},ensure_ascii=False)

    def edit(self):
        return {'path':'nir_eval/app.py','before':'return 1','after':'return 0'}

    def run_doctor(self,answers,advice=None,**kwargs):
        answers=iter(answers)
        with patch.object(doctor.OpenAICompatibleProvider,'generate',return_value=Generation(advice or self.advice(),{}, {'cost_usd':None})) as generate:
            code=doctor.run_doctor(profile=self.profile,input_fn=lambda _:next(answers),print_fn=self.logs.append,source_root=self.source,report_root=self.root/'reports',**kwargs)
        return code,generate

    def sources(self):
        return doctor.source_context(self.source,'CLI',lambda x:x)

    def test_full_diagnose_without_apply(self):
        code,generate=self.run_doctor(['Не работает CLI','да','нет'],self.advice([self.edit()]))
        self.assertEqual(code,0);generate.assert_called_once()
        self.assertEqual(self.path.read_text(),self.original)
        report=json.loads(next((self.root/'reports').glob('*/diagnosis.json')).read_text())
        self.assertFalse(report['applied']);self.assertEqual(report['validation'],'syntax_only_not_behavioral_tests')
        self.assertFalse(list((self.root/'reports').glob('*/backup')))

    def test_approve_apply_and_backup_exact_original(self):
        code,_=self.run_doctor(['Не работает CLI','да','да'],self.advice([self.edit()]))
        self.assertEqual(code,0);self.assertIn('return 0',self.path.read_text())
        backup=next((self.root/'reports').glob('*/backup/nir_eval/app.py'))
        self.assertEqual(backup.read_text(),self.original)
        report=json.loads(next((self.root/'reports').glob('*/diagnosis.json')).read_text())
        self.assertTrue(report['applied'])

    def test_no_provider_construction_without_consent_or_local_only(self):
        for local_only,answers in [(False,['CLI','нет']),(True,['CLI'])]:
            with self.subTest(local_only=local_only),patch.object(doctor,'OpenAICompatibleProvider') as provider:
                it=iter(answers)
                code=doctor.run_doctor(profile=self.profile,input_fn=lambda _:next(it),print_fn=self.logs.append,source_root=self.source,local_only=local_only)
                self.assertIn(code,[0,130]);provider.assert_not_called()

    def test_path_traversal_and_secret_module_rejected(self):
        for name in ['../outside.py','/tmp/outside.py','nir_eval/credentials.py']:
            edit={**self.edit(),'path':name}
            with self.subTest(name=name),self.assertRaises(ValidationError):
                doctor.prepare_repairs(self.source,self.sources(),[edit])
        self.assertEqual(self.path.read_text(),self.original)

    def test_invalid_python_rejected(self):
        with self.assertRaises(ValidationError):
            doctor.prepare_repairs(self.source,self.sources(),[{**self.edit(),'after':'return ('}])

    def test_stale_source_rejected_before_prepare_and_apply(self):
        sources=self.sources();self.path.write_text(self.original+'# another editor\n')
        with self.assertRaises(ValidationError):doctor.prepare_repairs(self.source,sources,[self.edit()])
        self.path.write_text(self.original)
        prepared=doctor.prepare_repairs(self.source,self.sources(),[self.edit()])
        self.path.write_text(self.original+'# another editor\n')
        with self.assertRaises(ValidationError):doctor.apply_repairs(prepared,self.root/'backups')
        self.assertFalse((self.root/'backups').exists())

    def test_symlink_source_excluded(self):
        self.path.unlink();target=self.root/'outside.py';target.write_text(self.original);self.path.symlink_to(target)
        self.assertEqual(self.sources(),{})

    def test_parent_symlink_swap_after_prepare_rejected(self):
        prepared=doctor.prepare_repairs(self.source,self.sources(),[self.edit()])
        parent=self.path.parent;parent.rename(self.source/'old-package')
        outside=self.root/'outside';outside.mkdir();(outside/'app.py').write_text(self.original)
        parent.symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValidationError):doctor.apply_repairs(prepared,self.root/'backup')
        self.assertEqual((outside/'app.py').read_text(),self.original)

    def test_secret_redaction_in_payload_model_output_and_report(self):
        code,generate=self.run_doctor(['Ошибка '+self.secret,'да'],self.advice(diagnosis='Ответ '+self.secret,steps=['Bearer '+self.secret]),last_error=self.secret)
        self.assertEqual(code,0)
        payload=generate.call_args.args[0][1]['content']
        self.assertNotIn(self.secret,payload)
        self.assertNotIn(self.secret,'\n'.join(self.logs))
        for file in (self.root/'reports').rglob('*'):
            if file.is_file():self.assertNotIn(self.secret,file.read_text())

    def test_source_containing_known_secret_never_sent(self):
        self.path.write_text(self.original+'# '+self.secret+'\n')
        code,generate=self.run_doctor(['CLI','да'])
        self.assertEqual(code,0)
        payload=json.loads(generate.call_args.args[0][1]['content'])
        self.assertEqual(payload['sources'],{})

    def test_malformed_model_response_does_not_write_source(self):
        code,_=self.run_doctor(['CLI','да'],'not json')
        self.assertEqual(code,2);self.assertEqual(self.path.read_text(),self.original)

    def test_backup_preserves_original_newline_bytes(self):
        original=b'def main():\r\n    return 1\r\n'
        self.path.write_bytes(original)
        prepared=doctor.prepare_repairs(self.source,self.sources(),[self.edit()])
        doctor.apply_repairs(prepared,self.root/'backup')
        self.assertEqual((self.root/'backup/nir_eval/app.py').read_bytes(),original)
        self.assertEqual(self.path.read_bytes(),original.replace(b'return 1',b'return 0'))

    def test_second_write_failure_restores_first_file(self):
        other=self.source/'nir_eval/interactive.py';other.write_text(self.original)
        edits=[self.edit(),{**self.edit(),'path':'nir_eval/interactive.py'}]
        prepared=doctor.prepare_repairs(self.source,self.sources(),edits)
        real_replace=os.replace
        calls=[]
        def failing_replace(source,target):
            calls.append(target)
            if len(calls)==2:raise OSError('fixture write error')
            return real_replace(source,target)
        with patch('nir_eval.doctor.os.replace',side_effect=failing_replace),self.assertRaises(OSError):
            doctor.apply_repairs(prepared,self.root/'backup')
        self.assertEqual(self.path.read_text(),self.original)
        self.assertEqual(other.read_text(),self.original)
        self.assertEqual((self.root/'backup/nir_eval/app.py').read_text(),self.original)

    def test_ambiguous_snippet_rejected(self):
        self.path.write_text(self.original+'\ndef other():\n    return 1\n')
        with self.assertRaises(ValidationError):doctor.prepare_repairs(self.source,self.sources(),[self.edit()])


if __name__=='__main__':unittest.main()
