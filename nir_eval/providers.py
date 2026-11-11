"""Provider adapters. Credentials are resolved from environment and never logged."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess


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
    """Compatibility API name; execution is exclusively through Promptfoo."""
    kind = 'model-inference'

    def __init__(self, config, allow_remote=False):
        from .promptfoo_backend import PromptfooProvider
        self.backend = PromptfooProvider(config, allow_remote)

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def generate(self, messages, model, parameters, context):
        return self.backend.generate(messages, model, parameters, context)


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
    if name == "openai-compatible":
        from .promptfoo_backend import PromptfooProvider
        return PromptfooProvider(config, allow_remote)
    if name == "replay": return ReplayProvider(config, allow_remote)
    if name in {"oracle-control", "invalid-control"}: return None
    raise ValueError(f"Unknown provider type: {name}")
