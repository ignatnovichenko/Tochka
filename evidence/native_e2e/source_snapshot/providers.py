"""Provider adapters. Credentials are resolved from environment and never logged."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
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


def decode_response(raw: str) -> Generation:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Generation(raw)
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        if data.get("error"):
            raise ProviderError(str(data["error"])[:500])
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
        self.config = config
        base_url = config.get("base_url", "").rstrip("/")
        parsed = urllib.parse.urlparse(base_url)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be a valid http(s) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Do not embed credentials, queries or fragments in base_url")
        if not local and parsed.scheme != "https":
            raise ValueError("Remote provider requires HTTPS")
        if not local and not allow_remote:
            raise ValueError("Remote provider requires --allow-remote after checking terms and cost")
        self.url = base_url + "/chat/completions"
        self.api_key = os.environ.get(config.get("api_key_env", "NIR_API_KEY"), "")
        # Refuse redirecting a bearer token or benchmark payload to another destination.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        self.opener = urllib.request.build_opener(NoRedirect())

    def generate(self, messages, model, parameters, context) -> Generation:
        blocked = {"messages", "model", "stream"} & parameters.keys()
        if blocked:
            raise ValueError("Reserved API parameter overridden")
        body = json.dumps({"model": model, "messages": messages,
                           "stream": False, **parameters}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with self.opener.open(req, timeout=float(self.config.get("timeout_seconds", 180))) as response:
                raw = response.read(1_000_001)
        except urllib.error.HTTPError as exc:
            # No error body, URL or authorization values are copied to records.
            raise ProviderError(f"Provider returned HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError):
            raise ProviderError("Provider connection failed or timed out") from None
        if len(raw) > 1_000_000:
            raise ProviderError("API response exceeds 1 MB limit")
        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Missing text response")
            return Generation(content, data.get("usage") or {},
                              {"response_model": data.get("model"),
                               "system_fingerprint": data.get("system_fingerprint"),
                               "finish_reason": data["choices"][0].get("finish_reason")})
        except (ValueError, KeyError, IndexError, TypeError):
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
    if name == "command": return CommandProvider(config, allow_remote)
    if name == "openai-compatible": return OpenAICompatibleProvider(config, allow_remote)
    if name == "replay": return ReplayProvider(config, allow_remote)
    if name == "transformers-local":
        from .local_model import TransformersLocalProvider
        return TransformersLocalProvider(config, allow_remote)
    if name in {"oracle-control", "invalid-control"}: return None
    raise ValueError(f"Unknown provider type: {name}")
