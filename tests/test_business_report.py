import copy
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

from nir_eval.business_report import terminal_summary, write_business_report
from nir_eval.business.schema import ValidationError


def candidate(**changes):
    data = {'model': 'model-a', 'configuration': 'baseline', 'attempts': 4,
            'judge_evaluated': 4, 'judge_unknown': 0, 'judge_coverage': 1,
            'mean_judge_score': .9, 'judge_pass_rate_among_evaluated': 1,
            'judge_critical_error_rate_among_evaluated': 0,
            'mean_solver_latency_seconds': 2.1, 'mean_total_latency_seconds': 5.2,
            'mean_cost_usd': .012, 'rejection_reasons': [], 'run': 'solver-baseline'}
    data.update(changes)
    return data


def selection(candidates=None, status='selected'):
    candidates = candidates or [candidate()]
    return {'status': status, 'selected': candidates[0] if status == 'selected' else None,
            'tied': candidates if status == 'tied_configurations' else [],
            'policy': {'min_mean_judge_score': .8, 'min_judge_coverage': 1, 'max_mean_cost_usd': .02},
            'candidates': candidates, 'sources': [], 'warnings': []}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def render(self, result=None, source=None):
        output = self.root / 'report'
        write_business_report(output, result or selection(), source_root=source)
        return (output / 'report.html').read_text(encoding='utf-8')
    def test_selected_and_unknown_cost_are_unambiguous(self):
        c = candidate(mean_cost_usd=None)
        text = self.render(selection([c]))
        self.assertIn('Под ваши требования подходит', text)
        self.assertIn('<strong>неизвестно</strong>', text)
        self.assertIn('Стоимость / задачу', text)
        self.assertNotIn('<strong>$0</strong>', text)
        summary = terminal_summary(selection([c]))
        self.assertIn('стоимость с проверкой: неизвестно', summary)
    def test_incomplete_quality_has_denominator_and_warning(self):
        c = candidate(judge_evaluated=2, judge_unknown=2, judge_coverage=.5,
                      mean_judge_score=1, rejection_reasons=['judge_coverage_below_threshold'])
        result = selection([c], status='no_eligible_configuration')
        text = self.render(result)
        self.assertIn('Подходящих конфигураций нет', text)
        self.assertIn('Оценено 2 из 4', text)
        self.assertIn('Неполная проверка', text)
        self.assertIn('Судья оценил недостаточно ответов', text)
        summary = terminal_summary(result)
        self.assertIn('100.0% по 2 из 4', summary)
        self.assertIn('НЕПОЛНАЯ ПРОВЕРКА', summary)
    def test_ties_are_not_arbitrarily_ranked(self):
        result = selection([candidate(), candidate(model='model-b', configuration='second')], 'tied_configurations')
        text = self.render(result)
        self.assertIn('Несколько подходящих вариантов', text)
        self.assertIn('second · model-b', text)
        self.assertIn('baseline · model-a', terminal_summary(result))
    def test_html_and_csv_injection_are_escaped(self):
        hostile = '<script>alert(1)</script>'
        c = candidate(model=hostile, configuration='=HYPERLINK("https://evil")', run='not-present')
        text = self.render(selection([c]))
        self.assertNotIn(hostile, text)
        self.assertIn('&lt;script&gt;', text)
        with (self.root/'report/comparison.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows[0]['configuration'].startswith("'="))
    def test_terminal_control_sequences_not_emitted(self):
        summary = terminal_summary(selection([candidate(model='x\x1b[31m\nspoof')]))
        self.assertNotIn('\x1b', summary)
        self.assertNotIn('\nspoof', summary)
    def populate_source(self):
        source = self.root/'source'
        record = {'id':'task-1','model':'model-a','configuration':'baseline','repetition':0,
                  'status':'passed','score':1,'total_cost_usd':.03,
                  'judge':{'status':'evaluated','score':1,'cost_usd':.02,'latency_seconds':3,
                           'criteria':[{'id':'accuracy','score':4,'rationale':'Correct <answer>',
                                        'evidence':[{'source':'candidate','quote':'five'}]}],
                           'critical_errors':[{'id':'wrong','detected':False,'rationale':'No error'}]}}
        write_rows(source/'solver-baseline/records.jsonl',[record])
        write_rows(source/'solver-baseline/responses.jsonl',[{**record,'response':'five <script>bad</script>'}])
        write_rows(source/'solver-baseline/solver_records.jsonl',[{**record,'latency_seconds':2,'provider_metadata':{'cost_usd':.01}}])
        write_rows(source/'inputs/cases.jsonl',[{'id':'task-1','task':'Give number five','context':'Business document','reference':'five'}])
        return source
    def test_reselect_report_embeds_answers_without_mutating_source(self):
        source = self.populate_source()
        before = {p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
        c = candidate(attempts=1,judge_evaluated=1,mean_cost_usd=.03)
        result = selection([c]); original = copy.deepcopy(result)
        text = self.render(result, source)
        self.assertIn('Give number five', text)
        self.assertIn('five &lt;script&gt;bad&lt;/script&gt;', text)
        self.assertIn('Correct &lt;answer&gt;', text)
        self.assertIn('Выполнение: $0.01', text)
        self.assertIn('Судья: $0.02', text)
        self.assertEqual(result, original)
        self.assertEqual(before,{p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()})
        class Links(HTMLParser):
            def handle_starttag(self, tag, attrs):
                if tag=='a':
                    href=dict(attrs).get('href')
                    self.assert_path(href)
        parser=Links();parser.assert_path=lambda href:self.assertTrue((self.root/'report'/href).is_file())
        parser.feed(text)
    def test_missing_sources_keep_summary_without_broken_links(self):
        text = self.render(source=self.root/'absent')
        self.assertIn('Подробные записи задач недоступны', text)
        self.assertNotIn('solver-baseline/report.html', text)
    def test_source_traversal_is_rejected_before_outputs(self):
        with self.assertRaises(ValidationError):
            self.render(selection([candidate(run='../private')]))
        self.assertFalse((self.root/'report/comparison.json').exists())
    def test_source_symlink_escape_is_rejected(self):
        source=self.root/'source'; source.mkdir()
        external=self.root/'external';external.mkdir()
        (source/'solver-baseline').symlink_to(external,target_is_directory=True)
        with self.assertRaises(ValidationError):self.render(source=source)
    def test_detail_unknown_score_is_not_rendered_as_zero(self):
        source=self.populate_source()
        p=source/'solver-baseline/records.jsonl'
        r=json.loads(p.read_text());r.update(status='evaluation_error',score=None)
        r['judge']={'status':'error','score':None,'reason_code':'judge_invalid_structured_output'}
        write_rows(p,[r])
        c=candidate(attempts=1,judge_evaluated=0,judge_coverage=0,mean_judge_score=None,mean_cost_usd=None)
        text=self.render(selection([c],'no_eligible_configuration'),source)
        self.assertIn('Этот ответ не включён в среднее качество',text)
        self.assertIn('Оценка судьи:</b> неизвестно',text)
    def test_zero_cost_is_preserved_as_known(self):
        c=candidate(mean_cost_usd=0,mean_solver_cost_usd=0,mean_judge_cost_usd=0)
        text=self.render(selection([c]))
        self.assertIn('<strong>$0</strong>',text)
        self.assertIn('Выполнение: $0',text)
        self.assertIn('Судья: $0',text)
    def test_current_saved_evidence_can_be_reported_offline(self):
        project=Path(__file__).resolve().parents[1]
        source=project/'runs/current_business'
        if not source.exists(): self.skipTest('Historical run not present')
        result=json.loads((source/'comparison.json').read_text())
        text=self.render(result,source)
        self.assertIn('Оценено 2 из 4',text)
        self.assertIn('supplier_delay_01',text)
        self.assertIn('Просим в кратчайшие сроки',text)
        self.assertEqual(json.loads((self.root/'report/comparison.json').read_text()),result)

if __name__ == '__main__':
    unittest.main()
