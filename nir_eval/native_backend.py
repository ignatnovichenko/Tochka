"""Tochka's Python API transport and bounded, streaming matrix scheduler.

The transport is restored from d97e79c. Concurrency and durable callback ordering
extend that transport without changing the strict judge or evidence formats.
"""
from __future__ import annotations
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from http.client import HTTPException
import json
import math
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid

from . import __version__
from .business.schema import ValidationError, strict_json
from .providers import ProviderError, ProviderUnavailable

_COOLDOWNS = {}
_COOLDOWN_LOCK = threading.Lock()
SCHEDULER = 'native.threadpool'
MAX_RESPONSE_BYTES = 1_000_000


def runtime_status():
    return {'ready': True, 'engine': 'native', 'version': __version__,
            'scheduler': SCHEDULER, 'cache': False, 'automatic_retries': 0,
            'dependencies': 'Python standard library; no Node.js or Promptfoo'}


def _unavailable(code, attempted=False, delay=None):
    error = ProviderUnavailable(code)
    error.request_attempted = attempted
    if delay is not None:
        error.retry_after_seconds = delay
    return error


class NativeProvider:
    kind = 'model-inference'

    def __init__(self, config, allow_remote=False):
        from .api_support import validate_api_config
        from .credentials import api_key
        local = validate_api_config(config, allow_remote=allow_remote, require_key=True)
        self.config = dict(config)
        self.api_key = api_key(config)
        self.session_id = str(uuid.uuid4())
        self.blocked_until = 0
        self.url = config['base_url'].rstrip('/') + '/chat/completions'
        self.cooldown_key = (config['base_url'].rstrip('/'), hashlib.sha256(self.api_key.encode()).hexdigest())
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        # No environment proxy may receive localhost business data or credentials.
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}) if local else urllib.request.ProxyHandler(), NoRedirect())

    def cooldown(self):
        with _COOLDOWN_LOCK:
            expiry = max(self.blocked_until, _COOLDOWNS.get(self.cooldown_key, 0))
        if expiry > time.time():
            raise _unavailable('api_cooldown_active', delay=expiry-time.time())

    def generate(self, messages, model, parameters, context):
        self.cooldown()
        if not isinstance(parameters, dict) or {'messages','model','stream','tools','tool_choice'} & parameters.keys():
            raise ValueError('Reserved API parameter overridden')
        body = json.dumps({'model':model, 'messages':messages, 'stream':False, **parameters}, allow_nan=False).encode()
        headers = {'Content-Type':'application/json', 'User-Agent':'Tochka/'+__version__,
                   'x-opencode-session':self.session_id}
        if self.api_key:
            headers['Authorization'] = 'Bearer '+self.api_key
        request = urllib.request.Request(self.url, data=body, headers=headers, method='POST')
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=float(self.config.get('timeout_seconds', 180))) as response:
                raw = response.read(MAX_RESPONSE_BYTES+1)
        except urllib.error.HTTPError as exc:
            from .api_support import api_error_code
            try:
                raw_error = exc.read(16384) if exc.code == 400 else b''
                code = api_error_code(exc.code, raw_error)
                error = (ProviderUnavailable if exc.code in {401,403,404,408,429,500,502,503,504} else ProviderError)(code)
                error.request_attempted = True
                if exc.code == 429:
                    retry = exc.headers.get('Retry-After') if exc.headers else None
                    try: delay = float(retry)
                    except (ValueError, TypeError):
                        try: delay = parsedate_to_datetime(retry).timestamp()-time.time()
                        except (ValueError,TypeError,AttributeError,OverflowError): delay = 60
                    if not math.isfinite(delay): delay = 60
                    error.retry_after_seconds = max(1,min(delay,7*86400))
                    self.blocked_until = time.time()+error.retry_after_seconds
                    with _COOLDOWN_LOCK:
                        _COOLDOWNS[self.cooldown_key] = max(_COOLDOWNS.get(self.cooldown_key,0),self.blocked_until)
            finally:
                exc.close()
            raise error from None
        except (OSError, HTTPException):
            raise _unavailable('api_connection_unavailable', attempted=True) from None
        if len(raw)>MAX_RESPONSE_BYTES:
            error = ProviderError('API response exceeds 1 MB limit')
            error.request_attempted = True
            raise error
        from .api_support import normalize_completion
        try:
            generation = normalize_completion(strict_json(raw), self.config, self.api_key)
        except (ValueError,TypeError):
            error = ProviderError('Malformed chat completion response')
            error.request_attempted = True
            raise error from None
        seconds = time.monotonic()-started
        generation.metadata.update(execution_engine='native', execution_engine_version=__version__,
                                   engine_wall_seconds=seconds, engine_call_seconds=seconds)
        return generation


def evaluate_jobs(jobs, max_concurrency=4, on_result=None, *, on_started=None):
    """Execute jobs once, stream callbacks on the caller thread, cap active calls.

    A start is acknowledged only after its journal callback finishes. A worker
    takes no next job until the previous completion is persisted. On interruption
    queued jobs stop immediately; already transmitted HTTP requests may still
    finish remotely. They are not retried or reported as free/unattempted.
    Daemon workers never keep an interrupted CLI alive waiting on a network timeout.
    """
    if type(max_concurrency) is not int or not 1<=max_concurrency<=32:
        raise ValidationError('Concurrency must be an integer from 1 to 32')
    if not isinstance(jobs,list) or len(jobs)>10000:
        raise ValidationError('A matrix supports at most 10000 pending jobs')
    if on_result is None:
        raise ValueError('Completion callback required')
    prepared=[]; identifiers=set(); payload_size=0
    # Validate the whole matrix and resolve all credentials before any HTTP call.
    for job in jobs:
        job_id=job.get('job_id')
        if not isinstance(job_id,str) or not job_id or len(job_id)>256 or job_id in identifiers:
            raise ValidationError('Unique nonempty job identifiers required')
        identifiers.add(job_id)
        parameters=job.get('parameters',{})
        if not isinstance(parameters,dict) or {'messages','model','stream','tools','tool_choice'} & parameters.keys():
            raise ValidationError('Reserved API parameter overridden')
        if not isinstance(job.get('model'),str) or not isinstance(job.get('messages'),list):
            raise ValidationError('Invalid matrix model/messages')
        try:
            payload_size+=len(json.dumps({'model':job['model'],'messages':job['messages'],'parameters':parameters},allow_nan=False).encode())
        except (ValueError,TypeError):
            raise ValidationError('Invalid JSON matrix payload') from None
        if payload_size>64_000_000:
            raise ValidationError('Matrix payload exceeds 64 MB; split the dataset')
        prepared.append((job,NativeProvider(job['provider'],job.get('allow_remote') is True)))
    summary={'scheduler':SCHEDULER,'completed':0,'test_count':len(jobs),'total_calls':0,'max_active_requests':0}
    if not jobs:return summary
    pending=queue.Queue(); events=queue.Queue(); stopped=threading.Event()
    active=0; counter_lock=threading.Lock()
    for item in prepared:pending.put(item)

    def emit(event):
        acknowledged=threading.Event()
        events.put((*event,acknowledged))
        while not acknowledged.wait(.05):
            if stopped.is_set():return False
        return not stopped.is_set()

    def worker():
        nonlocal active
        while not stopped.is_set():
            try:job,provider=pending.get_nowait()
            except queue.Empty:return
            job_id=job['job_id']; started=None; attempted=False
            try:
                provider.cooldown()
                started=datetime.now(timezone.utc).isoformat()
                if not emit(('started',job_id,started)):return
                with counter_lock:
                    active+=1
                    summary['max_active_requests']=max(summary['max_active_requests'],active)
                attempted=True
                tick=time.monotonic()
                try:
                    answer=provider.generate(job['messages'],job['model'],job.get('parameters',{}),job.get('context',{}))
                    answer.metadata.update(execution_scheduler=SCHEDULER,request_attempted=True,
                        latency_scope='Native API call; excludes scheduler queue wait',started_at_utc=started)
                except ProviderError as exc:
                    answer=exc
                    attempted=getattr(exc,'request_attempted',True)
                finally:
                    with counter_lock:active-=1
                seconds=time.monotonic()-tick
            except ProviderError as exc:
                answer=exc;seconds=0
            except BaseException:
                emit(('fatal',job_id,None));return
            if isinstance(answer,ProviderError):answer.request_attempted=attempted
            if attempted:
                with counter_lock:summary['total_calls']+=1
            if not emit(('result',job_id,(answer,seconds))):return

    threads=[threading.Thread(target=worker,name='tochka-api',daemon=True) for _ in range(min(max_concurrency,len(jobs)))]
    for thread in threads:thread.start()
    try:
        while summary['completed']<len(jobs):
            kind,job_id,value,acknowledged=events.get()
            if kind=='started':
                if on_started:on_started(job_id,value)
            elif kind=='result':
                on_result(job_id,*value)
                summary['completed']+=1
            else:
                raise ProviderUnavailable('native_scheduler_failed')
            acknowledged.set()
        return dict(summary)
    finally:
        stopped.set()
        for thread in threads:thread.join(timeout=.05)
