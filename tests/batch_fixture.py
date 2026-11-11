"""Synchronous fixture dispatcher for business tests; not an execution backend."""
from nir_eval.providers import Generation,ProviderUnavailable
from nir_eval.business.providers import JudgeUnavailable


def batch_fixture(solver,judge=None):
    def execute(jobs,max_concurrency,on_result,*,on_started=None):
        for job in jobs:
            if on_started:on_started(job['job_id'],'2026-11-11T12:00:00+00:00')
            try:
                if job['context'].get('config')=='business-judge':
                    reply=judge.judge(job['messages'],job['context']['id'])
                    result=Generation(reply.text,reply.usage,reply.metadata)
                else:
                    result=solver.generate(job['messages'],job['model'],job['parameters'],job['context'])
            except JudgeUnavailable as exc:
                result=ProviderUnavailable(str(exc).removeprefix('judge_'))
                if hasattr(exc,'retry_after_seconds'):result.retry_after_seconds=exc.retry_after_seconds
            except Exception as exc:result=exc
            on_result(job['job_id'],result,0.001)
        return {'scheduler':'fixture-not-promptfoo','completed':len(jobs),'max_concurrency':max_concurrency}
    return execute
