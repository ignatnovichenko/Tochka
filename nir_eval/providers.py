"""Provider adapters. Credentials are resolved from environment and never logged."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import time
import uuid
from email.utils import parsedate_to_datetime
import os
from pathlib import Path
import signal
import subprocess
import urllib.error
import urllib.parse
import urllib.request


@dataclass
class Generation:
    text: str
    usage: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class ProviderError(RuntimeError):
    pass


class ProviderUnavailable(ProviderError):
    """Explicit unavailable transport, never a semantic model failure."""
    pass


def decode_response(raw: str) -> Generation:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Generation(raw)
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        if data.get("error"):
            raise (ProviderUnavailable if data.get("unavailable") is True else ProviderError)("Command adapter reported unavailable" if data.get("unavailable") is True else "Command adapter reported an error")
        return Generation(data["text"], data.get("usage") or {}, data.get("metadata") or {})
    # A plain answer JSON is valid model text, not an adapter response.
    return Generation(raw)


class CommandProvider:
    kind = "model-inference"

    def __init__(self, config: dict, allow_remote: bool = False):
        self.config = config
        self.command = config.get("command")
        if not isinstance(self.command, list) or not self.command or not all(
                isinstance(x, str) for x in self.command):
            raise ValueError("command must be a nonempty argument list (never shell text)")
        if config.get("remote", False) and not allow_remote:
            raise ValueError("Remote command requires --allow-remote after checking terms and cost")

    def generate(self, messages, model, parameters, context) -> Generation:
        payload = {"messages": messages, "model": model, "parameters": parameters, **context}
        timeout = float(self.config.get("timeout_seconds", 180))
        proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True,
                                cwd=self.config.get("cwd"))
        try:
            stdout, stderr = proc.communicate(json.dumps(payload), timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise ProviderError(f"Command timed out after {timeout:g}s") from None
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
        if proc.returncode:
            # Do not include stderr: it may contain authentication data or private paths.
            raise ProviderError(f"Command exited with status {proc.returncode}; stderr not logged")
        if len(stdout) > 1_000_000:
            raise ProviderError("Command response exceeds 1 MB limit")
        return decode_response(stdout)


class OpenAICompatibleProvider:
    kind = "model-inference"

    def __init__(self, config: dict, allow_remote: bool = False):
        from .api_support import validate_api_config
        local = validate_api_config(config, allow_remote=allow_remote, require_key=True)
        self.config = config
        base_url = config['base_url'].rstrip('/')
        self.blocked_until = 0
        self.session_id = str(uuid.uuid4())
        self.url = base_url + "/chat/completions"
        from .credentials import api_key
        self.api_key = api_key(config)
        # Refuse redirecting a bearer token or benchmark payload to another destination.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}) if local else urllib.request.ProxyHandler(), NoRedirect())

    def generate(self, messages, model, parameters, context) -> Generation:
        if time.time() < self.blocked_until:
            exc = ProviderUnavailable('api_cooldown_active')
            exc.retry_after_seconds = max(0, self.blocked_until - time.time())
            raise exc
        blocked = {"messages", "model", "stream", "tools", "tool_choice"} & parameters.keys()
        if blocked:
            raise ValueError("Reserved API parameter overridden")
        body = json.dumps({"model": model, "messages": messages,
                           "stream": False, **parameters}).encode()
        headers = {"Content-Type": "application/json", "User-Agent": "Tochka/0.3.0", "x-opencode-session": self.session_id}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with self.opener.open(req, timeout=float(self.config.get("timeout_seconds", 180))) as response:
                raw = response.read(1_000_001)
        except urllib.error.HTTPError as exc:
            # No error body, URL or authorization values are copied to records.
            error = ProviderUnavailable(f'api_http_{exc.code}') if exc.code in {401,403,404,408,429,500,502,503,504} else ProviderError(f'api_http_{exc.code}')
            if exc.code == 429:
                retry = exc.headers.get('Retry-After') if exc.headers else None
                try:
                    delay = float(retry)
                except (ValueError, TypeError):
                    try: delay = parsedate_to_datetime(retry).timestamp() - time.time()
                    except (ValueError, TypeError, AttributeError, OverflowError): delay = 60
                import math
                if not math.isfinite(delay): delay = 60
                error.retry_after_seconds = max(1, min(delay, 7*86400))
                self.blocked_until = time.time() + error.retry_after_seconds
            exc.close()
            raise error from None
        except (urllib.error.URLError, TimeoutError):
            raise ProviderUnavailable('api_connection_unavailable') from None
        if len(raw) > 1_000_000:
            raise ProviderError("API response exceeds 1 MB limit")
        try:
            from .business.schema import strict_json
            data = strict_json(raw)
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Missing text response")
            from .api_support import response_metadata
            if self.api_key:
                content = content.replace(self.api_key, '[REDACTED_CREDENTIAL]')
            usage = data.get('usage') if isinstance(data.get('usage'),dict) else {}
            usage = {k:v for k,v in usage.items() if k in {'prompt_tokens','completion_tokens','total_tokens'} and type(v) is int and v>=0}
            meta = response_metadata(data, self.config)
            if self.api_key:
                def redact(value):
                    if isinstance(value,str): return value.replace(self.api_key,'[REDACTED_CREDENTIAL]')
                    if isinstance(value,dict): return {k:redact(v) for k,v in value.items()}
                    if isinstance(value,list): return [redact(v) for v in value]
                    return value
                meta = redact(meta)
            return Generation(content, usage, meta)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ProviderError("Malformed chat completion response") from None


class ReplayProvider:
    kind = "replay-not-inference"

    def __init__(self, config, allow_remote=False):
        self.rows = {}
        for line in Path(config["path"]).read_text().splitlines():
            row = json.loads(line)
            key = (row["id"], row["config"], row.get("repetition", 0))
            if key in self.rows:
                raise ValueError(f"Duplicate replay entry: {key}")
            self.rows[key] = row

    def generate(self, messages, model, parameters, context):
        key = (context["id"], context["config"], context["repetition"])
        row = self.rows.get(key)
        if row is None:
            raise ProviderError("Missing replay response")
        if row.get("error"):
            raise ProviderError("Original response contained a provider error")
        return Generation(row["raw_output"], row.get("token_usage") or {},
                          {"original_latency_seconds": row.get("latency_seconds"),
                           "original_model": row.get("model")})


def make_provider(config, allow_remote=False):
    name = config["type"]
    if name in {'opencode', 'command', 'transformers-local'}:
        class APIOnlyProvider:
            kind = 'model-inference'
            def generate(self, *args, **kwargs):
                raise ProviderError('api_only_legacy_transport_disabled')
        return APIOnlyProvider()
    if name == "openai-compatible": return OpenAICompatibleProvider(config, allow_remote)
    if name == "replay": return ReplayProvider(config, allow_remote)
    if name in {"oracle-control", "invalid-control"}: return None
    raise ValueError(f"Unknown provider type: {name}")
