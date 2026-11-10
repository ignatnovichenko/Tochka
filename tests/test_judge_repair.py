import json
import unittest
from nir_eval.business.evaluation import evaluate
from nir_eval.business.providers import BudgetedJudge, JudgeReply, JudgeUnavailable

CASE={'id':'task1','task':'Return five','context':'five','rubric_id':'r'}
RUBRIC={'id':'r','criteria':[{'id':'correct','description':'Answer five'}],'critical_errors':[]}
VALID=json.dumps({'case_id':'task1','criteria':[{'id':'correct','score':4,'rationale':'Correct','evidence':[{'source':'candidate','quote':'five'}]}],'critical_errors':[]})

class Sequence:
    model='test-judge';kind='model-inference'
    def __init__(self, replies):self.replies=list(replies);self.calls=0
    def judge(self,messages,case_id):
        self.calls+=1
        reply=self.replies.pop(0)
        if isinstance(reply,Exception):raise reply
        return JudgeReply(reply,{}, {'cost_usd':.01})

class RepairTests(unittest.TestCase):
    def test_repair_keeps_both_attempts_and_costs(self):
        seq=Sequence(['invalid',VALID]);judge=BudgetedJudge(seq,2,1)
        row=evaluate(CASE,RUBRIC,'five',judge)
        self.assertEqual(row['score'],1)
        self.assertEqual(seq.calls,2)
        self.assertEqual(len(row['judge']['attempts']),2)
        self.assertEqual(row['judge']['cost_usd'],.02)
        self.assertIsNotNone(row['judge']['attempts'][0]['validation_error'])
    def test_budget_includes_repair(self):
        seq=Sequence(['invalid',VALID]);judge=BudgetedJudge(seq,1,1)
        row=evaluate(CASE,RUBRIC,'five',judge)
        self.assertIsNone(row['score']);self.assertEqual(seq.calls,1)
        self.assertEqual(row['judge']['reason_code'],'judge_call_budget_exhausted')
    def test_no_network_retry(self):
        seq=Sequence([JudgeUnavailable('judge_http_429'),VALID]);judge=BudgetedJudge(seq,3,2)
        row=evaluate(CASE,RUBRIC,'five',judge)
        self.assertEqual(seq.calls,1);self.assertIsNone(row['score'])
    def test_repair_exhaustion_unknown_not_zero(self):
        seq=Sequence(['invalid','invalid',VALID]);judge=BudgetedJudge(seq,3,1)
        row=evaluate(CASE,RUBRIC,'five',judge)
        self.assertEqual(seq.calls,2);self.assertIsNone(row['score']);self.assertIsNone(row['passed'])
    def test_failed_second_call_cost_unknown(self):
        seq=Sequence(['invalid',JudgeUnavailable('judge_http_503')]);judge=BudgetedJudge(seq,2,1)
        row=evaluate(CASE,RUBRIC,'five',judge)
        self.assertIsNone(row['judge']['cost_usd'])
    def test_default_does_not_retry(self):
        seq=Sequence(['invalid',VALID]);judge=BudgetedJudge(seq,2,0)
        self.assertIsNone(evaluate(CASE,RUBRIC,'five',judge)['score']);self.assertEqual(seq.calls,1)

if __name__=='__main__':unittest.main()
