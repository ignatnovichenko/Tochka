"""Pinned Promptfoo execution; Tochka retains evaluation and accounting semantics."""
from __future__ import annotations
import atexit
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

VERSION = '0.125.1'
BUNDLED = Path(__file__).parent / 'promptfoo_runtime'
_WORKERS = weakref.WeakSet()
_VERIFIED = set()

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
    node = shutil.which('node')
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
    ready = compatible and installed == VERSION and (path / 'worker.mjs').is_file()
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
            'ready': ready,
            'runtime_path': str(path), 'telemetry': False, 'cache': False, 'automatic_retries': 0}


def require_runtime():
    status = runtime_status()
    if not status['ready']:
        raise ValidationError('Promptfoo не готов. Нужен Node.js >=22.22.0; выполните /setup или tochka setup.')
    return Path(status['runtime_path'])


def setup_runtime():
    if runtime_status()['ready']:
        return runtime_status()
    if not runtime_status()['node_compatible'] or not shutil.which('npm'):
        raise ValidationError('Установите Node.js >=22.22.0 и npm, затем выполните tochka setup.')
    target = install_path()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.exists():
        raise ValidationError('Неполная установка Promptfoo уже существует; сохраните/удалите каталог runtime и повторите setup.')
    stage = Path(tempfile.mkdtemp(prefix='.promptfoo-', dir=target.parent))
    try:
        for name in ('package.json', 'package-lock.json', 'worker.mjs'):
            shutil.copyfile(BUNDLED / name, stage / name)
        # npm output may include machine-specific proxy credentials; never copy it to reports.
        result = subprocess.run([shutil.which('npm'), 'ci', '--ignore-scripts', '--no-audit', '--no-fund'],
                                cwd=stage, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
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
        self.proc = subprocess.Popen([shutil.which('node'), str(self.path / 'worker.mjs')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, cwd=self.scratch.name, start_new_session=True)

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
        with self.lock:
            if self.proc is None:
                self.start()
            self.counter += 1
            request = dict(payload, id=self.counter)
            wire = (json.dumps(request, ensure_ascii=True, allow_nan=False) + '\n').encode()
            if len(wire) > 4_000_000:
                raise ProviderError('promptfoo_request_too_large')
            timeout = float(payload['timeout_seconds']) + 20  # bounded cold-start allowance
            deadline = time.monotonic() + timeout
            try:
                # Nonblocking IO also bounds a stalled worker before it consumes stdin.
                os.set_blocking(self.proc.stdin.fileno(), False)
                os.set_blocking(self.proc.stdout.fileno(), False)
                sent = 0
                out = bytearray()
                with selectors.DefaultSelector() as selector:
                    selector.register(self.proc.stdin, selectors.EVENT_WRITE)
                    selector.register(self.proc.stdout, selectors.EVENT_READ)
                    while True:
                        left = deadline - time.monotonic()
                        if left <= 0:
                            raise TimeoutError()
                        events = selector.select(left)
                        if not events:
                            raise TimeoutError()
                        for key, _ in events:
                            if key.fileobj is self.proc.stdin:
                                sent += os.write(key.fd, wire[sent:sent+65536])
                                if sent == len(wire): selector.unregister(self.proc.stdin)
                            else:
                                chunk = os.read(key.fd, 65536)
                                if not chunk: raise EOFError()
                                out.extend(chunk)
                                if len(out) > 2_000_000: raise ValueError()
                                if b'\n' in out:
                                    from .business.schema import strict_json
                                    result = strict_json(out)
                                    if not isinstance(result, dict) or result.get('id') != self.counter:
                                        raise ValueError()
                                    return result
            except (OSError, EOFError, ValueError, TimeoutError):
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
        if result.get('ok') is not True:
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
                                   engine_wall_seconds=time.monotonic() - started)
        latency = result.get('latency_ms')
        if type(latency) in (int, float) and math.isfinite(latency) and latency >= 0:
            generation.metadata['engine_call_seconds'] = latency / 1000
        return generation
