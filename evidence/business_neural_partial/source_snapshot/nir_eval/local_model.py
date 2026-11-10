"""Optional local Transformers backend, loaded once, with networking disabled."""
from pathlib import Path
import time
from .providers import Generation, ProviderError


class TransformersLocalProvider:
    kind = "model-inference"

    def __init__(self, config, allow_remote=False):
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ValueError("Install the optional local-model dependencies; see docs/LOCAL_MODELS_RU.md") from None
        self.torch = torch
        self.config = config
        model_path = Path(config["model_path"]).resolve()
        if not model_path.is_dir():
            raise ValueError("Local model_path does not exist; no automatic downloads are performed")
        threads = int(config.get("threads", 4))
        if not 1 <= threads <= 64:
            raise ValueError("threads must be between 1 and 64")
        torch.set_num_threads(threads)
        torch.manual_seed(int(config.get("seed", 42)))
        torch.use_deterministic_algorithms(True)
        dtype_name = config.get("dtype", "float32")
        if dtype_name not in {"float32", "bfloat16", "float16"}:
            raise ValueError("Unsupported model dtype")
        start = time.monotonic()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, dtype_name), attn_implementation="sdpa")
        self.model.eval()
        self.metadata = {"model_load_seconds": time.monotonic()-start, "torch_version": torch.__version__,
                         "transformers_version": transformers.__version__, "device": "cpu", "dtype": dtype_name,
                         "model_revision": config.get("revision", "unknown"),
                         "model_path": str(model_path), "threads": threads,
                         "parameter_count": sum(p.numel() for p in self.model.parameters()),
                         "base_generation_config": self.model.generation_config.to_dict()}

    def generate(self, messages, model, parameters, context):
        allowed = {"max_new_tokens", "do_sample", "temperature", "top_p", "top_k",
                   "repetition_penalty", "num_beams", "use_cache"}
        if parameters.keys() - allowed:
            raise ValueError(f"Unsupported local generation parameters: {sorted(parameters.keys()-allowed)}")
        params = {"do_sample": False, "max_new_tokens": 192, "num_beams": 1, "use_cache": True, **parameters}
        if not isinstance(params["max_new_tokens"], int) or not 1 <= params["max_new_tokens"] <= 4096:
            raise ValueError("max_new_tokens must be an integer between 1 and 4096")
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        # No silent context truncation: fail explicitly if the model window is exceeded.
        window = getattr(self.model.config, "max_position_embeddings", None)
        if window and inputs["input_ids"].shape[1] + params["max_new_tokens"] > window:
            raise ProviderError("Input plus requested output exceeds local model context window")
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **params, pad_token_id=self.tokenizer.eos_token_id)
        tokens = output[0, inputs["input_ids"].shape[1]:]
        return Generation(self.tokenizer.decode(tokens, skip_special_tokens=True),
                          {"prompt_tokens": inputs["input_ids"].shape[1], "completion_tokens": tokens.shape[0]},
                          {**self.metadata, "hit_token_limit": len(tokens) >= params["max_new_tokens"],
                           "effective_parameter_overrides": params})
