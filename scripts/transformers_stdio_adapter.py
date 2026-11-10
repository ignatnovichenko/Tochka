#!/usr/bin/env python3
"""One-request subprocess wrapper of the real built-in Transformers provider."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llama_cpp_stdio_adapter import diagnostics_to_stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        raw = sys.stdin.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("Input exceeds 1 MB")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("model", args.model_id) != args.model_id:
            raise ValueError("Requested model ID does not match actual local model")
        parameters = payload.get("parameters", {})
        if parameters.get("do_sample", False) is not False:
            raise ValueError("This smoke adapter supports greedy generation only")
        if parameters.get("temperature", 0) != 0:
            raise ValueError("This smoke adapter does not accept nonzero temperature")
        with diagnostics_to_stderr():
            from nir_eval.local_model import TransformersLocalProvider
            provider = TransformersLocalProvider({"model_path": args.model_path,
                "revision": args.model_revision, "threads": args.threads,
                "dtype": "float32", "seed": 42})
            result = provider.generate(payload["messages"], args.model_id, parameters,
                                       {k: payload[k] for k in ("id", "config", "repetition") if k in payload})
        result.metadata.update({"inference_type": "real_local_model_inference",
            "process_id": os.getpid(), "one_request_per_process": True,
            "retries": 0, "repair_calls": 0,
            "provider_timer_scope": "command startup, model load, tokenization, generation, decode and process teardown"})
        print(json.dumps({"text": result.text, "usage": result.usage,
                          "metadata": result.metadata}, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
