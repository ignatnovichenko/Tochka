"""Pinned Promptfoo execution; Tochka retains evaluation and accounting semantics."""
from __future__ import annotations
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import weakref
import uuid
from email.utils import parsedate_to_datetime

from .business.schema import ValidationError
from .providers import ProviderError, ProviderUnavailable
from .node_runtime import node_command, npm_command, command_environment

VERSION = '0.125.1'
BUNDLED = Path(__file__).parent / 'promptfoo_runtime'
_WORKERS = weakref.WeakSet()
_VERIFIED = set()
_COOLDOWNS = {}

def _close_workers():
    for worker in list(_WORKERS): worker.close()

atexit.register(_close_workers)


def install_path():
    return Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local/share') / 'tochka' / ('promptfoo-' + VERSION)


def runtime_path():
    override = os.environ.get('TOCHKA_PROMPTFOO_DIR')
    if override:
        return Path(override).expanduser().resolve()
    if (BUNDLED / 'node_modules/promptfoo/package.json').is_file():
        return BUNDLED
    return install_path()


def runtime_status():
    node = node_command()
    node_version = None
    compatible = False
    if node:
        try:
            node_version = subprocess.run([node, '--version'], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
            parts = tuple(int(x) for x in node_version.lstrip('v').split('.'))
            compatible = parts >= (22, 22, 0)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    path = runtime_path()
    installed = None
    try:
        installed = json.loads((path / 'node_modules/promptfoo/package.json').read_text())['version']
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        bridge_matches = ((path/'worker.mjs').read_bytes() == (BUNDLED/'worker.mjs').read_bytes()
                          and (path/'package-lock.json').read_bytes() == (BUNDLED/'package-lock.json').read_bytes())
    except OSError:
        bridge_matches = False
    ready = compatible and installed == VERSION and bridge_matches
    if ready:
        fingerprint = (str(path.resolve()), node_version, (path / 'worker.mjs').stat().st_mtime_ns,
                       (path / 'node_modules/promptfoo/package.json').stat().st_mtime_ns)
        if fingerprint not in _VERIFIED:
            worker = Worker(path)
            try:
                health = worker.call({'op': 'health', 'timeout_seconds': 10})
                ready = health.get('ok') is True and health.get('version') == VERSION
                if ready: _VERIFIED.add(fingerprint)
            except (ProviderError, OSError):
                ready = False
            finally:
                worker.close()
    return {'engine': 'promptfoo', 'version_required': VERSION, 'version_installed': installed,
            'node_version': node_version, 'node_compatible': compatible,
            'ready': ready, 'bridge_matches': bridge_matches,
            'runtime_path': str(path), 'telemetry': False, 'cache': False, 'automatic_retries': 0}


def require_runtime():
    status = runtime_status()
    if not status['ready']:
        raise ValidationError('Promptfoo не готов. Нужен Node.js >=22.22.0; выполните /setup или tochka setup.')
    return Path(status['runtime_path'])


def setup_runtime():
    if runtime_status()['ready']:
        return runtime_status()
    if not runtime_status()['node_compatible'] or not npm_command():
        raise ValidationError('Установите Node.js >=22.22.0 и npm, затем выполните tochka setup.')
    target = install_path()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.environ.get('TOCHKA_PROMPTFOO_DIR'):
        raise ValidationError('Явный TOCHKA_PROMPTFOO_DIR не готов; исправьте этот runtime или уберите переменную перед setup.')
    if target.exists():
        # A Tochka update can change the bridge without changing Promptfoo itself.
        # Reuse verified pinned dependencies; atomically replace only our own worker.
        try:
            installed = json.loads((target/'node_modules/promptfoo/package.json').read_text())['version']
            same_lock = (target/'package-lock.json').read_bytes() == (BUNDLED/'package-lock.json').read_bytes()
            if installed != VERSION or not same_lock or target.is_symlink(): raise ValueError()
            old_worker = (target/'worker.mjs').read_bytes()
        except (OSError, ValueError, KeyError, TypeError):
            raise ValidationError('Установка Promptfoo повреждена; сохраните/удалите каталог runtime и повторите setup.') from None
        fd, name = tempfile.mkstemp(prefix='.worker-', dir=target)
        try:
            with os.fdopen(fd,'wb') as stream: stream.write((BUNDLED/'worker.mjs').read_bytes())
            os.replace(name,target/'worker.mjs')
            result = runtime_status()
            if result['ready']: return result
            (target/'worker.mjs').write_bytes(old_worker)
            raise ValidationError('Проверка обновлённого runtime не пройдена; прежний worker восстановлен.')
        finally:
            if os.path.exists(name): os.unlink(name)

    stage = Path(tempfile.mkdtemp(prefix='.promptfoo-', dir=target.parent))
    try:
        for name in ('package.json', 'package-lock.json', 'worker.mjs'):
            shutil.copyfile(BUNDLED / name, stage / name)
        # npm output may include machine-specific proxy credentials; never copy it to reports.
        result = subprocess.run([npm_command(), 'ci', '--ignore-scripts', '--no-audit', '--no-fund'],
                                cwd=stage, env=command_environment(os.environ), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
        if result.returncode:
            raise ValidationError('npm ci завершился ошибкой; проверьте доступ к registry.npmjs.org и повторите setup.')
        stage.rename(target)
    except subprocess.TimeoutExpired:
        raise ValidationError('Установка Promptfoo превысила 15 минут; повторите setup.') from None
    finally:
        if stage.exists(): shutil.rmtree(stage)
    result = runtime_status()
    if not result['ready']:
        raise ValidationError('Promptfoo установлен, но проверка запуска не пройдена. Проверьте Node.js и повторите tochka setup --check.')
    return result


class Worker:
    """One sequential JSON-lines worker, no key in argv, disk, or logs."""
    def __init__(self, path):
        self.path = path
        self.proc = None
        self.lock = threading.Lock()
        self.counter = 0
        self.scratch = None
        _WORKERS.add(self)

    def __del__(self):
        try: self.close()
        except Exception: pass

    def start(self):
        self.scratch = tempfile.TemporaryDirectory(prefix='tochka-promptfoo-')
        env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'LANG', 'TZ', 'SSL_CERT_FILE', 'NODE_EXTRA_CA_CERTS') if k in os.environ}
        env.update(HOME=self.scratch.name, XDG_CONFIG_HOME=self.scratch.name, XDG_CACHE_HOME=self.scratch.name,
                   PROMPTFOO_CONFIG_DIR=self.scratch.name, PROMPTFOO_DISABLE_TELEMETRY='1',
                   PROMPTFOO_DISABLE_UPDATE='1', PROMPTFOO_CACHE_ENABLED='false',
                   PROMPTFOO_LOG_LEVEL='error', PROMPTFOO_DISABLE_FILE_LOGGING='1')
        self.proc = subprocess.Popen([node_command(), str(self.path / 'worker.mjs')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=command_environment(env), cwd=self.scratch.name, start_new_session=True)

    def close(self):
        proc, self.proc = self.proc, None
        if proc:
            if proc.poll() is None:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            proc.wait()
            for stream in (proc.stdin, proc.stdout):
                if stream: stream.close()
        if self.scratch:
            self.scratch.cleanup(); self.scratch = None

    def call(self, payload):
        return self.exchange(payload)

    def exchange(self, payload, on_event=None):
        """Consume bounded NDJSON events without buffering a whole evaluation."""
        with self.lock:
            if self.proc is None:
                self.start()
            self.counter += 1
            request = dict(payload, id=self.counter)
            wire = (json.dumps(request, ensure_ascii=True, allow_nan=False) + '\n').encode()
            if len(wire) > 64_000_000:
                raise ProviderError('promptfoo_request_too_large')
            deadline = time.monotonic() + float(payload['timeout_seconds']) + 20
            try:
                os.set_blocking(self.proc.stdin.fileno(), False)
                os.set_blocking(self.proc.stdout.fileno(), False)
                sent = 0
                out = bytearray()
                with selectors.DefaultSelector() as selector:
                    selector.register(self.proc.stdin, selectors.EVENT_WRITE)
                    selector.register(self.proc.stdout, selectors.EVENT_READ)
                    while True:
                        left = deadline - time.monotonic()
                        if left <= 0: raise ProviderUnavailable('promptfoo_worker_timeout')
                        events = selector.select(left)
                        if not events: raise ProviderUnavailable('promptfoo_worker_timeout')
                        for key, _ in events:
                            if key.fileobj is self.proc.stdin:
                                sent += os.write(key.fd, wire[sent:sent+65536])
                                if sent == len(wire): selector.unregister(self.proc.stdin)
                            else:
                                chunk = os.read(key.fd, 65536)
                                if not chunk: raise ProviderUnavailable('promptfoo_worker_unavailable')
                                out.extend(chunk)
                                while b'\n' in out:
                                    line, _, rest = out.partition(b'\n'); out = bytearray(rest)
                                    from .business.schema import strict_json
                                    try: result = strict_json(line)
                                    except (ValueError, TypeError):
                                        raise ProviderUnavailable('promptfoo_protocol_error') from None
                                    if len(line)>2_097_152 or not isinstance(result,dict) or result.get('id')!=self.counter:
                                        raise ProviderUnavailable('promptfoo_protocol_error')
                                    if result.get('event') in {'started','result'}:
                                        if on_event is None: raise ProviderUnavailable('promptfoo_protocol_error')
                                        on_event(result)
                                    else:
                                        return result
                                if len(out)>2_097_152: raise ProviderUnavailable('promptfoo_protocol_error')
            except (OSError, EOFError, TimeoutError):
                self.close()
                raise ProviderUnavailable('promptfoo_worker_unavailable') from None
            except BaseException:
                self.close()
                raise


class PromptfooProvider:
    kind = 'model-inference'
    def __init__(self, config, allow_remote=False):
        from .api_support import validate_api_config
        from .credentials import api_key
        validate_api_config(config, allow_remote=allow_remote, require_key=True)
        path = require_runtime()
        self.config = config
        self.api_key = api_key(config)
        self.session_id = str(uuid.uuid4())
        self.blocked_until = 0
        self.url = config['base_url'].rstrip('/') + '/chat/completions'
        self.worker = Worker(path)

    def generate(self, messages, model, parameters, context):
        if time.time() < self.blocked_until:
            exc = ProviderUnavailable('api_cooldown_active')
            exc.retry_after_seconds = max(0, self.blocked_until - time.time())
            raise exc
        if {'messages', 'model', 'stream', 'tools', 'tool_choice'} & parameters.keys():
            raise ValueError('Reserved API parameter overridden')
        started = time.monotonic()
        result = self.worker.call({'base_url': self.config['base_url'], 'api_key': self.api_key,
            'messages': messages, 'model': model, 'parameters': parameters,
            'timeout_seconds': self.config.get('timeout_seconds', 180), 'session_id': self.session_id})
        return self.decode_result(result, time.monotonic() - started)

    def decode_result(self, result, wall_seconds):
        if result.get('ok') is not True:
            if result.get('error') == 'api_cooldown_active':
                error = ProviderUnavailable('api_cooldown_active')
                delay = result.get('retry_after_seconds', 60)
                error.retry_after_seconds = delay if type(delay) in (int,float) and math.isfinite(delay) and delay>=0 else 60
                raise error
            status = result.get('status')
            if type(status) is int and 100 <= status <= 599:
                cls = ProviderUnavailable if status in {401,403,404,408,429,500,502,503,504} else ProviderError
                code = 'api_http_' + str(status)
                if status == 400 and result.get('error') in {'api_http_400_region_restricted','api_http_400_training_restricted'}:
                    code = result['error']
                error = cls(code)
                if status == 429:
                    retry = result.get('retry_after')
                    try: delay = float(retry)
                    except (ValueError, TypeError):
                        try: delay = parsedate_to_datetime(retry).timestamp() - time.time()
                        except (ValueError, TypeError, AttributeError, OverflowError): delay = 60
                    if not math.isfinite(delay): delay = 60
                    error.retry_after_seconds = max(1, min(delay, 7*86400))
                    self.blocked_until = time.time() + error.retry_after_seconds
                raise error
            raise ProviderUnavailable('promptfoo_transport_unavailable')
        if result.get('engine') != 'promptfoo' or result.get('version') != VERSION:
            raise ProviderError('promptfoo_protocol_mismatch')
        from .business.schema import strict_json
        try:
            data = strict_json(result['raw_response'])
        except (ValueError, TypeError, KeyError):
            raise ProviderError('Malformed chat completion response') from None
        from .api_support import normalize_completion
        generation = normalize_completion(data, self.config, self.api_key)
        generation.metadata.update(execution_engine='promptfoo', execution_engine_version=VERSION,
                                   engine_wall_seconds=wall_seconds)
        latency = result.get('latency_ms')
        if type(latency) in (int, float) and math.isfinite(latency) and latency >= 0:
            generation.metadata['engine_call_seconds'] = latency / 1000
        return generation


def evaluate_jobs(jobs, max_concurrency=4, on_result=None, *, on_started=None):
    """Execute an explicit matrix through Promptfoo.evaluate, streaming completions.

    Python builds/validates jobs and writes evidence; Promptfoo owns dispatch and
    concurrency. A fatal scheduler/worker failure interrupts the run, never fabricates
    completions. Finished callbacks remain durable for explicit resume.
    """
    from .api_support import validate_api_config
    from .credentials import api_key
    from .business.schema import canonical
    if type(max_concurrency) is not int or not 1 <= max_concurrency <= 32:
        raise ValidationError('Concurrency must be an integer from 1 to 32')
    if not isinstance(jobs, list) or len(jobs)>10000:
        raise ValidationError('A Promptfoo matrix supports at most 10000 pending jobs')
    if on_result is None: raise ValueError('Completion callback required')
    if not jobs:
        return {'scheduler':'promptfoo.evaluate','completed':0,'test_count':0,'total_calls':0,'max_active_requests':0}
    identities = set()
    providers = {}
    encoded = []
    contexts = {}
    cooldown_keys = {}
    for key, expiry in list(_COOLDOWNS.items()):
        if expiry <= time.time(): _COOLDOWNS.pop(key,None)
    for job in jobs:
        job_id = job.get('job_id')
        if not isinstance(job_id,str) or not job_id or len(job_id)>256 or job_id in identities:
            raise ValidationError('Unique nonempty job identifiers required')
        identities.add(job_id)
        config = job['provider']
        validate_api_config(config, allow_remote=job.get('allow_remote') is True, require_key=True)
        parameters = job.get('parameters', {})
        if not isinstance(parameters,dict) or {'messages','model','stream','tools','tool_choice'} & parameters.keys():
            raise ValidationError('Reserved API parameter overridden')
        if not isinstance(job.get('model'),str) or not isinstance(job.get('messages'),list):
            raise ValidationError('Invalid matrix model/messages')
        identity = canonical({'provider':config,'model':job['model'],'parameters':parameters})
        if identity not in providers:
            # A normalization context, not another worker or scheduling layer.
            provider = object.__new__(PromptfooProvider)
            provider.config = config
            provider.api_key = api_key(config)
            provider.blocked_until = 0
            provider.session_id = str(uuid.uuid4())
            providers[identity] = (f'p{len(providers)}', provider)
        provider_id, provider = providers[identity]
        cooldown_key = (config['base_url'].rstrip('/'), hashlib.sha256(provider.api_key.encode()).hexdigest())
        cooldown_keys[job_id] = cooldown_key
        encoded.append({'job_id':job_id, 'provider_id':provider_id, 'base_url':config['base_url'],
                        'api_key':provider.api_key, 'session_id':provider.session_id,
                        'messages':job['messages'], 'model':job['model'], 'parameters':parameters,
                        'timeout_seconds':config.get('timeout_seconds',180),
                        'blocked_until_ms':_COOLDOWNS.get(cooldown_key,0)*1000})
        contexts[job_id] = provider
    # Serialize and cap before starting a child process or sending any model request.
    timeout = sum(float(j['timeout_seconds']) for j in encoded) + 60
    payload = {'op':'evaluate','jobs':encoded,'max_concurrency':max_concurrency,'timeout_seconds':timeout}
    if len(json.dumps(payload,ensure_ascii=True,allow_nan=False).encode()) > 63_999_000:
        raise ValidationError('Matrix payload exceeds 64 MB; split the dataset')
    worker = Worker(require_runtime())
    finished = set()
    started = set()
    def event(message):
        job_id = message.get('job_id')
        if job_id not in contexts: raise ProviderUnavailable('promptfoo_unknown_job')
        if message['event'] == 'started':
            if job_id in started or job_id in finished: raise ProviderUnavailable('promptfoo_duplicate_job')
            started.add(job_id)
            if on_started: on_started(job_id, message.get('started_at_utc'))
            return
        if job_id in finished: raise ProviderUnavailable('promptfoo_duplicate_job')
        result = message.get('result')
        latency = message.get('latency_seconds')
        if not isinstance(result,dict) or type(latency) not in (int,float) or not math.isfinite(latency) or latency<0:
            raise ProviderUnavailable('promptfoo_protocol_error')
        if result.get('attempted') is not (job_id in started):
            raise ProviderUnavailable('promptfoo_protocol_error')
        try:
            answer = contexts[job_id].decode_result(result, latency)
            answer.metadata.update(execution_scheduler='promptfoo.evaluate',request_attempted=True,
                latency_scope='Promptfoo provider call; excludes scheduler queue and worker startup',
                started_at_utc=result.get('started_at_utc'))
        except ProviderError as exc:
            answer = exc
            delay = getattr(exc,'retry_after_seconds',None)
            if type(delay) in (int,float) and math.isfinite(delay) and delay > 0:
                key = cooldown_keys[job_id]
                _COOLDOWNS[key] = max(_COOLDOWNS.get(key,0),time.time()+min(delay,7*86400))
            answer.request_attempted = result.get('attempted') is True
        finished.add(job_id)
        on_result(job_id, answer, latency)
    try:
        summary = worker.exchange(payload, event)
        if (summary.get('ok') is not True or summary.get('scheduler')!='promptfoo.evaluate'
                or summary.get('version')!=VERSION or summary.get('completed')!=len(jobs)
                or finished != identities):
            raise ProviderUnavailable('promptfoo_matrix_incomplete')
        return {key:summary[key] for key in ('scheduler','completed','test_count','total_calls','max_active_requests')}
    finally:
        worker.close()
