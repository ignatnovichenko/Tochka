import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from unittest.mock import Mock, patch

from nir_eval.datasets import load_finqa, select_cases, file_sha256
from nir_eval.prompts import build_messages
from nir_eval.scoring import answers_equal, normalize_answer, parse_object, execute_program, score_output, program_tokens
from nir_eval.providers import make_provider, OpenAICompatibleProvider, CommandProvider, decode_response, ProviderError
from nir_eval.reporting import summarize, write_reports
from nir_eval.runner import run, load_config, export_plan, read_records
from nir_eval.importing import import_results

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/synthetic_smoke.json"


class DatasetTests(unittest.TestCase):
    def test_load_and_selection(self):
        cases = load_finqa(DATA)
        self.assertEqual(len(cases), 3)
        self.assertEqual(select_cases(cases, 2), select_cases(list(reversed(cases)), 2))
        with self.assertRaises(ValueError): select_cases(cases, 0)
        with self.assertRaises(ValueError): select_cases(cases, ids=["missing"])
        with self.assertRaises(ValueError): select_cases(cases, ids=[cases[0].id] * 2)

    def test_prompt_has_only_public_context(self):
        case = load_finqa(DATA)[0]
        messages = build_messages(case, "direct")
        self.assertNotIn('"exe_ans"', json.dumps(messages))
        self.assertNotIn('"expected_program"', json.dumps(messages))
        self.assertNotIn(case.expected_program, json.dumps(messages))
        self.assertNotIn(case.id, json.dumps(messages))
        with self.assertRaises(ValueError): build_messages(case, "gold")

    def test_duplicate_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            rows = json.loads(DATA.read_text())
            p.write_text(json.dumps(rows + [rows[0]]))
            with self.assertRaises(ValueError): load_finqa(p)

    def test_plan_never_contains_references(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = load_config(ROOT / "configs/smoke_oracle.json")
            result = export_plan(cfg, Path(d) / "plan.jsonl")
            self.assertEqual(result["attempts"], 18)
            self.assertFalse(result["model_inference_performed"])
            self.assertNotIn("expected_answer", (Path(d) / "plan.jsonl").read_text())


class ScoringTests(unittest.TestCase):
    def test_answers(self):
        self.assertTrue(answers_equal("14.464%", .14464))
        self.assertTrue(answers_equal(1/3, .33333))
        self.assertFalse(answers_equal("14%", .14464))
        self.assertFalse(answers_equal(True, 1))
        self.assertTrue(answers_equal(" YES ", "yes"))
        for v in [float('nan'), float('inf'), "Infinity", [], "USD 12", None]:
            with self.assertRaises(ValueError): normalize_answer(v)

    def test_json_strict(self):
        self.assertEqual(parse_object('```json\n{"answer":1}\n```')["answer"], 1)
        for text in ['The answer is {"answer":1}', '{"answer":1}{"answer":2}',
                     '{"answer": NaN}', '{"answer":1,"answer":2}', '[1]', "```bad"]:
            with self.assertRaises(ValueError): parse_object(text)

    def test_programs(self):
        self.assertEqual(execute_program("divide(20, 80), multiply(#0, const_100)", []), 25)
        self.assertEqual(execute_program("greater(5, 2)", []), "yes")
        self.assertEqual(execute_program("add(const_m1, 2)", []), 1)
        self.assertEqual(execute_program("table_average(amount, none)", [["amount", "$10", "20"]]), 15)
        self.assertEqual(program_tokens("add(1, 2)"), ["add(", "1", "2", ")", "EOF"])
        for program in ["__import__(os, 2)", "divide(1, 0)", "exp(1, 1001)",
                        "add(#0, 1)", "add(1, 2)junk", "add(1,2);add(3,4)",
                        "exp(-1, 0.5)", "table_sum(missing, none)", ""]:
            with self.assertRaises((ValueError, ZeroDivisionError)): execute_program(program, [])

    def test_scoring_protocols(self):
        for protocol, text in [('direct','{"answer":20}'), ('cot','{"reasoning":"100-80","answer":20}'),
                               ('program','{"program":"subtract(100, 80)"}')]:
            result = score_output(text, protocol, 20, "subtract(100, 80)", [])
            self.assertTrue(result.correct)
            self.assertIsNone(result.parse_error)
        self.assertIsNotNone(score_output("garbage", "direct", 20, "subtract(100, 80)", []).parse_error)


class ProviderTests(unittest.TestCase):
    def test_remote_guard(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleProvider({"base_url": "https://example.org/v1"})
        with self.assertRaises(ValueError):
            OpenAICompatibleProvider({"base_url": "http://example.org/v1"}, True)
        with self.assertRaises(ValueError):
            OpenAICompatibleProvider({"base_url": "https://user:secret@example.org"}, True)
        local = OpenAICompatibleProvider({"base_url": "http://127.0.0.1:8000/v1"})
        self.assertEqual(local.config['base_url'], 'http://127.0.0.1:8000/v1')

    def test_command_without_shell(self):
        p = CommandProvider({"command": ["python", "-c", "import sys,json;p=json.load(sys.stdin);print(json.dumps({'text':json.dumps({'answer':20}),'usage':{'input_tokens':7}}))"]})
        response = p.generate([], 'test', {}, {"id":"id"})
        self.assertEqual(response.usage["input_tokens"], 7)
        self.assertEqual(json.loads(response.text), {"answer":20})
        with self.assertRaises(ValueError): CommandProvider({"command":"echo hi"})
        with self.assertRaises(ValueError): CommandProvider({"command":["echo"], "remote": True})

    def test_errors_and_timeouts(self):
        p = CommandProvider({"command":["python","-c","import sys;print('secret',file=sys.stderr);sys.exit(3)"]})
        with self.assertRaises(ProviderError) as caught: p.generate([],"m",{}, {})
        self.assertNotIn('secret', str(caught.exception))
        p = CommandProvider({"command":["python","-c","import time;time.sleep(2)"],"timeout_seconds":.02})
        with self.assertRaises(ProviderError): p.generate([],"m",{}, {})

    def test_decode_plain_json(self):
        self.assertEqual(decode_response('{"answer":2}').text, '{"answer":2}')
        self.assertEqual(decode_response('plain text').text, 'plain text')

    def test_replay_missing_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'replay.jsonl'
            row = {"id":"x", "config":"direct", "repetition":0, "raw_output":'{"answer":1}'}
            p.write_text(json.dumps(row) + '\n')
            replay = make_provider({"type":"replay", "path":str(p)})
            self.assertEqual(replay.kind, 'replay-not-inference')
            self.assertEqual(replay.generate([],"m",{},row).text, row['raw_output'])
            with self.assertRaises(ProviderError): replay.generate([],"m",{},{**row,"id":"z"})
            p.write_text((json.dumps(row) + '\n') * 2)
            with self.assertRaises(ValueError): make_provider({"type":"replay", "path":str(p)})


class RunnerTests(unittest.TestCase):
    def run_silent(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()): return run(*args, **kwargs)

    def test_oracle_invalid_resume_and_reports(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = load_config(ROOT / "configs/smoke_oracle.json")
            out = Path(d) / 'oracle'
            summary = self.run_silent(cfg, out)
            self.assertEqual(len(read_records(out / 'records.jsonl')), 18)
            self.assertTrue(all(g['answer_accuracy'] == 1 for g in summary['groups']))
            self.assertTrue(all(g['provenance'] == 'control-not-llm' for g in summary['groups']))
            self.run_silent(cfg, out, resume=True)
            self.assertEqual(len(read_records(out / 'records.jsonl')), 18)
            with self.assertRaises(ValueError): self.run_silent(cfg, out)
            changed = copy.deepcopy(cfg); changed['seed'] = 50
            with self.assertRaises(ValueError): self.run_silent(changed, out, resume=True)
            for name in ['summary.json','summary.csv','records.csv','report.html','report.md','manifest.json']:
                self.assertTrue((out / name).is_file())
            cfg['provider']['type'] = 'invalid-control'
            summary = self.run_silent(cfg, Path(d) / 'invalid')
            self.assertTrue(all(g['answer_accuracy'] == 0 for g in summary['groups']))
            self.assertTrue(all(g['parse_errors'] == 6 for g in summary['groups']))

    def test_error_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = load_config(ROOT / 'configs/smoke_oracle.json')
            cfg['provider'] = {"type":"command", "command":["nonexistent-nir-executable"]}
            summary = self.run_silent(cfg, Path(d) / 'errors')
            self.assertEqual(sum(g['provider_errors'] for g in summary['groups']), 18)
            self.assertTrue(all(g['answer_accuracy'] == 0 for g in summary['groups']))

    def test_lock_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); (out / '.run.lock').write_text('active')
            with self.assertRaises(ValueError): self.run_silent(load_config(ROOT / 'configs/smoke_oracle.json'), out)
            bad = out / 'bad.jsonl'; bad.write_text('{}\n{')
            with self.assertRaises(ValueError): read_records(bad)

    def test_secret_config_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'config.json'
            p.write_text(json.dumps({'provider':{'api_key':'never-store-me'}}))
            with self.assertRaises(ValueError): load_config(p)


class ReportingTests(unittest.TestCase):
    def test_ci_clusters_tasks(self):
        rows = []
        for case in range(3):
            for repetition in range(4):
                rows.append(dict(id=str(case), provider='p', model='m', config='direct',
                    provenance='model-inference', correct=(case != 0), repetition=repetition,
                    parsed_answer=1, error=None, parse_error=None, latency_seconds=1, token_usage={}))
        g = summarize(rows)['groups'][0]
        self.assertEqual(g['unique_tasks'], 3)
        self.assertEqual(g['attempts'], 12)
        self.assertEqual(g['answer_accuracy'], 2/3)
        self.assertEqual(g['valid_answer_stability'], 1)
        self.assertIsNone(g['cost_usd'])
        self.assertIsNone(g['ttft_seconds'])

    def test_html_escape(self):
        with tempfile.TemporaryDirectory() as d:
            row = dict(id='<script>alert(1)</script>', provider='p',model='m',config='direct',
                provenance='model-inference',correct=False,parsed_answer=None,expected_answer=0,
                error=None,parse_error='<script>',latency_seconds=None,token_usage={})
            write_reports([row], d)
            report = (Path(d) / 'report.html').read_text()
            self.assertNotIn('<script>', report)
            self.assertIn('&lt;script&gt;', report)


class ImportTests(unittest.TestCase):
    def test_complete_verified_import(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cases = load_finqa(DATA)
            manifest = {'inference_type':'real_local_model_inference','dataset_sha256':file_sha256(DATA),
                        'n_cases':3,'protocols':['direct'],'repetitions':1,'model_id':'test-model',
                        'model_revision':'test-revision','completed_at_utc':'2026-11-10T00:00:00+00:00'}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'case_ids.json').write_text(json.dumps([c.id for c in cases]))
            rows = [dict(id=c.id, config='direct', repetition=0, model='test-model',model_revision='test-revision',
                         provider='fixture-not-real-model',messages=build_messages(c,'direct'),raw_output=json.dumps({'answer':c.expected_answer}),
                         latency_seconds=1.23,error=None,token_usage={'input_tokens':2}) for c in cases]
            p = root / 'records.jsonl'
            p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            summary = import_results(p, root / 'manifest.json', DATA, root / 'scored')
            self.assertEqual(summary['groups'][0]['answer_accuracy'], 1)
            self.assertEqual(summary['groups'][0]['latency_median_seconds'], 1.23)
            p.write_text(json.dumps(rows[0]) + '\n')
            with self.assertRaises(ValueError): import_results(p, root / 'manifest.json', DATA, root / 'incomplete')



class HTTPAdapterTests(unittest.TestCase):
    def test_request_and_usage(self):
        provider = OpenAICompatibleProvider({"base_url": "http://localhost:8000/v1"})
        data = {"choices": [{"message": {"content": '{"answer": 2}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9}, "model": "served-version"}
        provider.opener.open = Mock(return_value=io.BytesIO(json.dumps(data).encode()))
        messages = [{"role": "user", "content": "test"}]
        response = provider.generate(messages, 'test-model', {"temperature": 0}, {})
        self.assertEqual(response.usage['prompt_tokens'], 9)
        request = json.loads(provider.opener.open.call_args.args[0].data)
        self.assertEqual(request['model'], 'test-model')
        self.assertEqual(request['messages'], messages)
        self.assertEqual(request['temperature'], 0)
        self.assertIs(request['stream'], False)
        self.assertEqual(response.metadata['response_model'], 'served-version')
        with self.assertRaises(ValueError): provider.generate([], 'm', {"model": "override"}, {})

    def test_http_error_body_is_not_logged(self):
        provider = OpenAICompatibleProvider({"base_url": "http://localhost:8000/v1"})
        provider.opener.open = Mock(side_effect=HTTPError('http://localhost:8000/v1/chat/completions', 401, 'fixture', {}, io.BytesIO(b'PRIVATE_SECRET')))
        with self.assertRaises(ProviderError) as result:
            provider.generate([], 'm', {}, {})
        self.assertEqual(str(result.exception), 'api_http_401')


class ComparisonTests(unittest.TestCase):
    def test_rejects_controls_and_missing_runs(self):
        from nir_eval.comparing import compare_runs
        with self.assertRaises(ValueError): compare_runs([], 'unused')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'manifest.json').write_text(json.dumps({'status':'completed','provenance':'control-not-llm'}))
            with self.assertRaises(ValueError): compare_runs([root,root], root/'out')

    def test_same_scope_comparison_and_duplicate_guard(self):
        from nir_eval.comparing import compare_runs
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['a','b']:
                p=root/name;p.mkdir()
                manifest={'status':'completed','provenance':'model-inference','dataset_sha256':'fixture',
                          'case_ids':['x'],'prompt_version':'fixture','planned_attempts':1,'completed_attempts':1}
                (p/'manifest.json').write_text(json.dumps(manifest))
                row=dict(id='x',provider='fixture',model=name,config='direct',repetition=0,
                         provenance='model-inference',correct=(name=='a'),parsed_answer=1,
                         expected_answer=1,error=None,parse_error=None,latency_seconds=1,token_usage={})
                (p/'records.jsonl').write_text(json.dumps(row)+'\n')
            result=compare_runs([root/'a',root/'b'],root/'report')
            self.assertEqual(len(result['groups']),2)
            self.assertEqual(result['paired_comparisons'][0]['paired_tasks'],1)
            with self.assertRaises(ValueError): compare_runs([root/'a',root/'a'],root/'duplicate')

if __name__ == '__main__': unittest.main()
