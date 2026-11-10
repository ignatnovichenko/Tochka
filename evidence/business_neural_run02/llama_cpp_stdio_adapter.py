#!/usr/bin/env python3
"""One-request local Qwen GGUF adapter for nir_eval's command provider.

Reads one JSON object from stdin, performs one greedy completion, writes one
JSON envelope to stdout, then unloads the model. No network, sampling, repairs,
or retries. Native diagnostics are redirected to stderr, including C stdout.
Requires the already installed official llama-cpp-python == 0.3.16.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-GGUF"
MODEL_REVISION = "bb5d59e06d9551d752d08b292a50eb208b07ab1f"


def normalized_parameters(parameters):
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be an object")
    supported = {"max_tokens", "max_new_tokens", "temperature", "do_sample",
                 "top_p", "top_k", "repeat_penalty", "repetition_penalty", "seed"}
    if set(parameters) - supported:
        raise ValueError("Unsupported parameters: " + ", ".join(sorted(set(parameters) - supported)))
    if "do_sample" in parameters and parameters["do_sample"] is not False:
        raise ValueError("Only do_sample=false is supported")
    if parameters.get("temperature", 0.0) != 0.0:
        raise ValueError("Only temperature=0 is supported")
    if parameters.get("top_p", 1.0) != 1.0 or parameters.get("top_k", 0) != 0:
        raise ValueError("Greedy adapter requires top_p=1 and top_k=0")
    if "max_tokens" in parameters and "max_new_tokens" in parameters and parameters["max_tokens"] != parameters["max_new_tokens"]:
        raise ValueError("Conflicting max_tokens/max_new_tokens")
    max_tokens = parameters.get("max_tokens", parameters.get("max_new_tokens", 192))
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 2048:
        raise ValueError("max_tokens must be an integer in [1, 2048]")
    if "repeat_penalty" in parameters and "repetition_penalty" in parameters and parameters["repeat_penalty"] != parameters["repetition_penalty"]:
        raise ValueError("Conflicting repeat_penalty/repetition_penalty")
    penalty = parameters.get("repeat_penalty", parameters.get("repetition_penalty", 1.1))
    if isinstance(penalty, bool) or not isinstance(penalty, (float, int)) or not math.isfinite(penalty) or not 0.1 <= penalty <= 3:
        raise ValueError("repeat_penalty must be finite and in [0.1, 3]")
    seed = parameters.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2147483647:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    return {"max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
            "top_k": 0, "repeat_penalty": float(penalty), "seed": seed,
            "echo": False, "stop": ["<|im_end|>", "<|endoftext|>"]}


def render_chatml(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str):
            raise ValueError("Only system/user/assistant text messages are supported")
    return "".join("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"


@contextlib.contextmanager
def diagnostics_to_stderr():
    """Keep the command-adapter stdout channel valid even for native logging."""
    sys.stdout.flush()
    original = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(original, 1)
        os.close(original)


def generate(payload, args):
    if not isinstance(payload, dict):
        raise ValueError("stdin must contain an object")
    if payload.get("model", args.model_id) != args.model_id:
        raise ValueError("Requested model ID does not match this adapter's actual model")
    parameters = normalized_parameters(payload.get("parameters", {}))
    prompt = render_chatml(payload.get("messages"))
    model_path = Path(args.model_path).resolve(strict=True)
    if not model_path.is_file() or model_path.suffix != ".gguf":
        raise ValueError("--model-path must be an existing local GGUF file")
    import llama_cpp
    if llama_cpp.__version__ != "0.3.16":
        raise RuntimeError("This pinned adapter requires llama-cpp-python 0.3.16")
    start = time.perf_counter()
    model = llama_cpp.Llama(model_path=str(model_path), n_ctx=args.context,
                           n_threads=args.threads, n_threads_batch=args.threads,
                           n_batch=512, n_ubatch=512, n_gpu_layers=0,
                           seed=parameters["seed"], use_mmap=True, verbose=False)
    load_seconds = time.perf_counter() - start
    try:
        prompt_tokens = len(model.tokenize(prompt.encode(), add_bos=False, special=True))
        if prompt_tokens + parameters["max_tokens"] > args.context:
            raise ValueError("Prompt plus output budget exceeds context; refusing truncation")
        model.reset()
        start = time.perf_counter()
        response = model.create_completion(prompt, **parameters)
        inference_seconds = time.perf_counter() - start
        choice = response["choices"][0]
        metadata = {
            "inference_type": "real_local_model_inference",
            "engine": "llama-cpp-python", "engine_version": llama_cpp.__version__,
            "model_id": args.model_id, "model_revision": args.model_revision,
            "quantization": "Q4_K_M", "device": "cpu", "n_gpu_layers": 0,
            "n_threads": args.threads, "n_threads_batch": args.threads,
            "n_ctx": args.context, "n_batch": 512, "n_ubatch": 512,
            "use_mmap": True, "chat_template": "ChatML",
            "effective_generation_config": parameters,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_tokens_prechecked": prompt_tokens, "inputs_truncated": False,
            "model_loading_seconds": load_seconds,
            "completion_seconds": inference_seconds,
            "completion_timer_scope": "entire create_completion; excludes model load and teardown",
            "provider_timer_scope": "parent command provider includes process startup, model load, completion and teardown",
            "finish_reason": choice.get("finish_reason"),
            "hit_token_limit": choice.get("finish_reason") == "length",
            "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "process_id": os.getpid(), "retries": 0, "repair_calls": 0,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        manifest = model_path.parent / "model_manifest.json"
        if manifest.exists():
            metadata["weights_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            metadata["weights_verification_note"] = "Preverified local weights; this invocation records manifest hash but does not rehash multi-GB shards."
        return {"text": choice["text"], "usage": response.get("usage", {}), "metadata": metadata}
    finally:
        model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--context", type=int, default=4096)
    args = parser.parse_args()
    try:
        if not 1 <= args.threads <= 64 or not 256 <= args.context <= 32768:
            raise ValueError("Invalid threads/context")
        raw = sys.stdin.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("Input exceeds 1 MB")
        payload = json.loads(raw)
        with diagnostics_to_stderr():
            response = generate(payload, args)
        print(json.dumps(response, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
