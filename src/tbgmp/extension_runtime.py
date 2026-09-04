"""Repository-owned runtime used by the formal extension experiments.

This module preserves the API of the XEC/SIP experiment helper while keeping
model, context, and output paths configurable. Heavy GPU dependencies are
loaded lazily so configuration and policy checks can run on CPU-only hosts.
"""

from __future__ import annotations

import gc
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path


MODEL_REGISTRY = {
    "qwen3_4b": {
        "model_name": "Qwen3-4B-Instruct-2507",
        "preferred_dtype": "bfloat16",
        "family": "Qwen3",
        "scale": "4B",
        "prompt_mode": "chat_template",
        "decoding_mode": "deterministic",
        "max_new_tokens": 24,
    },
    "qwen25_3b_instruct": {
        "model_name": "Qwen2.5-3B-Instruct",
        "preferred_dtype": "bfloat16",
        "family": "Qwen2.5",
        "scale": "3B",
        "prompt_mode": "chat_template",
        "decoding_mode": "deterministic",
        "max_new_tokens": 24,
    },
}

SEEDS = {0: "AURORA-7749", 1: "NEBULA-3186"}
QUESTION = "What is the secret project code name? Answer with just the code name."
MAX_NEW_TOKENS = 24

FIELDS = [
    "model_id", "model_name", "family", "scale", "comparison_target",
    "trust_remote_code", "base_model_id", "experiment_variant", "prompt_mode",
    "decoding_mode", "max_new_tokens", "domain", "context_length",
    "actual_context_length", "needle_depth", "seed", "needle_string",
    "policy_name", "policy_type", "default_key_bits", "default_value_bits",
    "protected_key_bits", "protected_value_bits", "protected_key_layers",
    "protected_value_layers", "residual_window", "found", "accuracy",
    "response", "fp16_kv_mb", "compressed_kv_mb", "kv_compression_ratio",
    "kv_saving_percent", "peak_gpu_gb", "runtime_s", "tok_per_s", "error",
    "oom", "compatibility_error", "completed", "started_at", "finished_at",
    "notes", "stage", "case_type",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_chat_template(tokenizer, messages) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        joined = "\n\n".join(f"{item['role']}: {item['content']}" for item in messages)
        return f"{joined}\nassistant:"


def get_dims(model):
    config = getattr(model.config, "text_config", model.config)
    n_layers = int(config.num_hidden_layers)
    n_heads = int(config.num_attention_heads)
    n_kv_heads = int(
        getattr(config, "num_key_value_heads", getattr(config, "multi_query_group_num", n_heads))
    )
    hidden_size = int(config.hidden_size)
    head_dim = int(getattr(config, "head_dim", hidden_size // n_heads))
    return n_layers, n_heads, n_kv_heads, head_dim, hidden_size


def compressed_tensor_bytes(seq_len: int, n_kv_heads: int, head_dim: int, bits: int) -> int:
    if seq_len <= 0:
        return 0
    indices_per_byte = max(1, 8 // int(bits))
    return (
        seq_len * n_kv_heads * math.ceil(head_dim / indices_per_byte)
        + seq_len * n_kv_heads * 2
    )


def estimate_kv_bytes(actual_tokens: int, dims, policy: dict):
    n_layers, _, n_kv_heads, head_dim, _ = dims
    fp16 = n_layers * n_kv_heads * actual_tokens * head_dim * 2 * 2
    if policy.get("fp16"):
        return fp16, fp16, 1.0
    residual_window = int(policy.get("residual_window", 128))
    old_tokens = max(0, actual_tokens - residual_window)
    recent_tokens = min(actual_tokens, residual_window)
    compressed = 0
    for layer in range(n_layers):
        key_bits = int(policy.get("layer_key_bits", {}).get(layer, policy["default_key_bits"]))
        value_bits = int(policy.get("layer_value_bits", {}).get(layer, policy["default_value_bits"]))
        compressed += compressed_tensor_bytes(old_tokens, n_kv_heads, head_dim, key_bits)
        compressed += compressed_tensor_bytes(old_tokens, n_kv_heads, head_dim, value_bits)
        compressed += recent_tokens * n_kv_heads * head_dim * 2 * 2
    return fp16, compressed, fp16 / compressed if compressed else 0.0


def tbgmp_policy(model_id: str, k: int, ranked_layers: list[int], case_type: str) -> dict:
    default_key_bits = 4 if case_type == "K4-sensitive" else 2
    layers = [int(layer) for layer in ranked_layers[:k]]
    return {
        "policy_name": f"{model_id}_tbgmp_top{k}_keys",
        "policy_type": f"tbgmp_top{k}",
        "fp16": False,
        "default_key_bits": default_key_bits,
        "default_value_bits": 2,
        "protected_key_bits": 6,
        "protected_value_bits": 2,
        "protected_key_layers": layers,
        "protected_value_layers": [],
        "layer_key_bits": {layer: 6 for layer in layers},
        "layer_value_bits": {},
        "residual_window": 128,
        "notes": f"T-BGMP top-{k} key protection from calibration risk ranking.",
    }


def control_policy(
    model_id: str,
    kind: str,
    k: int,
    ranked_layers: list[int],
    case_type: str,
    seed: int | None = None,
) -> dict:
    if kind == "random":
        layers = sorted(random.Random(int(seed)).sample(range(len(ranked_layers)), k))
        name = f"{model_id}_tbgmp_random{k}_keys_seed{seed}"
        policy_type = f"random{k}"
    elif kind == "bottom":
        layers = [int(layer) for layer in ranked_layers[-k:]]
        name = f"{model_id}_tbgmp_bottom{k}_keys"
        policy_type = f"bottom{k}"
    else:
        raise ValueError(kind)
    default_key_bits = 4 if case_type == "K4-sensitive" else 2
    return {
        "policy_name": name,
        "policy_type": policy_type,
        "fp16": False,
        "default_key_bits": default_key_bits,
        "default_value_bits": 2,
        "protected_key_bits": 6,
        "protected_value_bits": 2,
        "protected_key_layers": layers,
        "protected_value_layers": [],
        "layer_key_bits": {layer: 6 for layer in layers},
        "layer_value_bits": {},
        "residual_window": 128,
        "notes": f"Same-budget {policy_type} key protection control.",
    }


class ExtensionRuntime:
    """Compatibility facade recovered from the formal HPC helper."""

    MODEL_REGISTRY = MODEL_REGISTRY
    SEEDS = SEEDS
    FIELDS = FIELDS

    def __init__(self, turboquant_root: str | Path):
        self.turboquant_root = Path(turboquant_root).expanduser().resolve()
        if not self.turboquant_root.is_dir():
            raise RuntimeError(f"TurboQuant root does not exist: {self.turboquant_root}")
        root_text = str(self.turboquant_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        self.DOMAIN_FILES: dict[str, str] = {}
        self._cache_class = None

    @staticmethod
    def get_dims(model):
        return get_dims(model)

    @staticmethod
    def estimate_kv_bytes(actual_tokens, dims, policy):
        return estimate_kv_bytes(actual_tokens, dims, policy)

    @staticmethod
    def tbgmp_policy(model_id, k, ranked_layers, case_type):
        return tbgmp_policy(model_id, k, ranked_layers, case_type)

    @staticmethod
    def control_policy(model_id, kind, k, ranked_layers, case_type, seed=None):
        return control_policy(model_id, kind, k, ranked_layers, case_type, seed)

    def _runtime_imports(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
            from turboquant.compressors_v3 import TurboQuantV3
        except ImportError as exc:
            raise RuntimeError(
                "Extension execution requires torch, transformers, and the configured TurboQuant runtime."
            ) from exc
        return torch, AutoModelForCausalLM, AutoTokenizer, DynamicCache, TurboQuantV3

    def _adaptive_cache_class(self):
        if self._cache_class is not None:
            return self._cache_class
        torch, _, _, DynamicCache, TurboQuantV3 = self._runtime_imports()

        class TBGMPAdaptiveCache(DynamicCache):
            def __init__(
                self,
                default_key_bits=2,
                default_value_bits=2,
                layer_key_bits=None,
                layer_value_bits=None,
                residual_window=128,
                n_layers=28,
                seed=42,
            ):
                super().__init__()
                self.default_key_bits = int(default_key_bits)
                self.default_value_bits = int(default_value_bits)
                self.layer_key_bits = {
                    int(key): int(value) for key, value in (layer_key_bits or {}).items()
                }
                self.layer_value_bits = {
                    int(key): int(value) for key, value in (layer_value_bits or {}).items()
                }
                self.residual_window = int(residual_window)
                self.n_layers = int(n_layers)
                self.seed = int(seed)
                self._compressors = {}
                self._chunks_k = {}
                self._chunks_v = {}
                self._fp16_recent_k = {}
                self._fp16_recent_v = {}
                self._total_seq = {}

            def _get_compressor(self, layer_idx, head_dim, device):
                key_bits = self.layer_key_bits.get(int(layer_idx), self.default_key_bits)
                value_bits = self.layer_value_bits.get(int(layer_idx), self.default_value_bits)
                cache_key = (int(layer_idx), int(head_dim), str(device), key_bits, value_bits)
                if cache_key not in self._compressors:
                    self._compressors[cache_key] = TurboQuantV3(
                        head_dim=head_dim,
                        key_bits=key_bits,
                        value_bits=value_bits,
                        residual_window=0,
                        layer_idx=layer_idx,
                        n_layers=self.n_layers,
                        protected_layers=0,
                        protected_bits=8,
                        seed=self.seed,
                        device=str(device),
                    )
                return self._compressors[cache_key]

            def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
                _, _, sequence_new, head_dim = key_states.shape
                compressor = self._get_compressor(layer_idx, head_dim, key_states.device)
                if layer_idx not in self._chunks_k:
                    self._chunks_k[layer_idx] = []
                    self._chunks_v[layer_idx] = []
                    self._fp16_recent_k[layer_idx] = []
                    self._fp16_recent_v[layer_idx] = []
                    self._total_seq[layer_idx] = 0
                self._total_seq[layer_idx] += int(sequence_new)
                self._fp16_recent_k[layer_idx].append(key_states)
                self._fp16_recent_v[layer_idx].append(value_states)
                recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
                recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
                overflow = (
                    recent_k.shape[2]
                    if self.residual_window == 0
                    else max(0, recent_k.shape[2] - self.residual_window)
                )
                if overflow > 0:
                    compressed_k, compressed_v = compressor.compress_kv(
                        recent_k[:, :, :overflow, :], recent_v[:, :, :overflow, :]
                    )
                    self._chunks_k[layer_idx].append(compressed_k)
                    self._chunks_v[layer_idx].append(compressed_v)
                    recent_k = recent_k[:, :, overflow:, :]
                    recent_v = recent_v[:, :, overflow:, :]
                    self._fp16_recent_k[layer_idx] = [recent_k] if recent_k.shape[2] else []
                    self._fp16_recent_v[layer_idx] = [recent_v] if recent_v.shape[2] else []
                key_parts, value_parts = [], []
                for compressed_k, compressed_v in zip(
                    self._chunks_k[layer_idx], self._chunks_v[layer_idx]
                ):
                    restored_k, restored_v = compressor.decompress_kv(compressed_k, compressed_v)
                    key_parts.append(restored_k.to(key_states.dtype))
                    value_parts.append(restored_v.to(value_states.dtype))
                if self._fp16_recent_k[layer_idx]:
                    key_parts.append(torch.cat(self._fp16_recent_k[layer_idx], dim=2))
                    value_parts.append(torch.cat(self._fp16_recent_v[layer_idx], dim=2))
                full_k = torch.cat(key_parts, dim=2) if key_parts else key_states
                full_v = torch.cat(value_parts, dim=2) if value_parts else value_states
                while len(self.layers) <= layer_idx:
                    from transformers.cache_utils import DynamicLayer

                    self.layers.append(DynamicLayer())
                return full_k, full_v

            def get_seq_length(self, layer_idx=0):
                return self._total_seq.get(layer_idx, 0)

        self._cache_class = TBGMPAdaptiveCache
        return self._cache_class

    def cache_from_policy(self, policy: dict, dims):
        if policy.get("fp16"):
            return None
        cache_class = self._adaptive_cache_class()
        return cache_class(
            default_key_bits=policy["default_key_bits"],
            default_value_bits=policy["default_value_bits"],
            layer_key_bits=policy.get("layer_key_bits", {}),
            layer_value_bits=policy.get("layer_value_bits", {}),
            residual_window=policy.get("residual_window", 128),
            n_layers=dims[0],
            seed=42,
        )

    def load_model(self, model_path: str, model_id: str):
        torch, AutoModelForCausalLM, AutoTokenizer, _, _ = self._runtime_imports()
        config = self.MODEL_REGISTRY[model_id]
        trust_remote_code = bool(config.get("trust_remote_code", True))
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
            use_fast=config.get("tokenizer_use_fast", True),
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if config.get("preferred_dtype") == "bfloat16" else torch.float16
        kwargs = {
            "device_map": "auto",
            "trust_remote_code": trust_remote_code,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, **kwargs
            )
        model.eval()
        return tokenizer, model

    def build_prompt(
        self,
        tokenizer,
        model_id: str,
        domain: str,
        context_length: int,
        needle_depth: int,
        seed: int,
        needle_string: str,
    ) -> str:
        source = Path(self.DOMAIN_FILES[domain]).read_text(encoding="utf-8", errors="ignore")
        source_tokens = tokenizer.encode(source, add_special_tokens=False)
        rng = random.Random(int(seed))
        if source_tokens:
            offset = rng.randrange(0, min(len(source_tokens), 997))
            source_tokens = source_tokens[offset:] + source_tokens[:offset]
        if len(source_tokens) < context_length:
            source_tokens *= context_length // max(len(source_tokens), 1) + 2
        needle = f"The secret project code name is {needle_string}."
        needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
        budget = max(1, int(context_length) - len(needle_tokens) - 64)
        position = max(0.0, min(1.0, float(needle_depth) / 100.0))
        before_length = int(budget * position)
        before = tokenizer.decode(source_tokens[:before_length], skip_special_tokens=True)
        after = tokenizer.decode(
            source_tokens[before_length : before_length + budget - before_length],
            skip_special_tokens=True,
        )
        document = (
            f"{before}\n\n--- Internal Memo ---\n{needle}\n"
            f"--- End Memo ---\n\n{after}"
        )
        return safe_chat_template(
            tokenizer,
            [
                {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
                {"role": "user", "content": f"Read this document:\n\n{document}\n\n{QUESTION}"},
            ],
        )

    def base_row(
        self, model_id, model_name, domain, context_length, depth, seed,
        needle_string, policy, stage, case_type,
    ) -> dict:
        config = self.MODEL_REGISTRY[model_id]
        return {
            "model_id": model_id,
            "model_name": model_name,
            "family": config.get("family", ""),
            "scale": config.get("scale", ""),
            "comparison_target": config.get("comparison_target", ""),
            "trust_remote_code": bool(config.get("trust_remote_code", True)),
            "base_model_id": config.get("base_model_id", model_id),
            "experiment_variant": config.get("experiment_variant", "standard"),
            "prompt_mode": config.get("prompt_mode", "chat_template"),
            "decoding_mode": config.get("decoding_mode", "deterministic"),
            "max_new_tokens": config.get("max_new_tokens", MAX_NEW_TOKENS),
            "domain": domain,
            "context_length": context_length,
            "actual_context_length": "",
            "needle_depth": depth,
            "seed": seed,
            "needle_string": needle_string,
            "policy_name": policy["policy_name"],
            "policy_type": policy.get("policy_type", ""),
            "default_key_bits": policy.get("default_key_bits", ""),
            "default_value_bits": policy.get("default_value_bits", ""),
            "protected_key_bits": policy.get("protected_key_bits", ""),
            "protected_value_bits": policy.get("protected_value_bits", ""),
            "protected_key_layers": json.dumps(policy.get("protected_key_layers", [])),
            "protected_value_layers": json.dumps(policy.get("protected_value_layers", [])),
            "residual_window": policy.get("residual_window", ""),
            "found": False,
            "accuracy": 0.0,
            "response": "",
            "fp16_kv_mb": "",
            "compressed_kv_mb": "",
            "kv_compression_ratio": "",
            "kv_saving_percent": "",
            "peak_gpu_gb": "",
            "runtime_s": "",
            "tok_per_s": "",
            "error": "",
            "oom": False,
            "compatibility_error": False,
            "completed": False,
            "started_at": now_iso(),
            "finished_at": "",
            "notes": policy.get("notes", ""),
            "stage": stage,
            "case_type": case_type,
        }

    def run_generation(
        self, model, tokenizer, dims, model_id, model_name, domain,
        context_length, depth, seed, needle_string, policy, stage, case_type,
    ) -> dict:
        torch, _, _, _, _ = self._runtime_imports()
        row = self.base_row(
            model_id, model_name, domain, context_length, depth, seed,
            needle_string, policy, stage, case_type,
        )
        try:
            prompt = self.build_prompt(
                tokenizer, model_id, domain, context_length, depth, seed, needle_string
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            actual_tokens = int(input_ids.shape[1])
            cache = self.cache_from_policy(policy, dims)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(self.MODEL_REGISTRY[model_id].get("max_new_tokens", MAX_NEW_TOKENS)),
                    do_sample=False,
                    use_cache=True,
                    past_key_values=cache,
                    pad_token_id=tokenizer.eos_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            runtime = time.perf_counter() - started
            new_tokens = generated[0][input_ids.shape[1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).replace("\n", " ").strip()
            found = needle_string.lower() in response.lower()
            fp16, compressed, ratio = self.estimate_kv_bytes(actual_tokens, dims, policy)
            row.update({
                "actual_context_length": actual_tokens,
                "found": bool(found),
                "accuracy": 1.0 if found else 0.0,
                "response": response,
                "fp16_kv_mb": round(fp16 / 1024**2, 4),
                "compressed_kv_mb": round(compressed / 1024**2, 4),
                "kv_compression_ratio": round(ratio, 4),
                "kv_saving_percent": round((1 - compressed / fp16) * 100, 4) if fp16 else 0.0,
                "peak_gpu_gb": (
                    round(torch.cuda.max_memory_allocated() / 1024**3, 4)
                    if torch.cuda.is_available() else ""
                ),
                "runtime_s": round(runtime, 4),
                "tok_per_s": round(len(new_tokens) / runtime if runtime > 0 else 0.0, 4),
            })
        except Exception as exc:
            message = f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ').strip()}"
            row["error"] = message
            row["oom"] = "out of memory" in message.lower() or (
                "cuda" in message.lower() and "memory" in message.lower()
            )
            row["compatibility_error"] = bool(not policy.get("fp16") and not row["oom"])
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            row["completed"] = True
            row["finished_at"] = now_iso()
        return row


def create_extension_runtime(turboquant_root: str | Path) -> ExtensionRuntime:
    return ExtensionRuntime(turboquant_root)
