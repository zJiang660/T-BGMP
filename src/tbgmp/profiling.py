from __future__ import annotations

import importlib
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .prompting import apply_chat_template
from .risk_score import compute_risk_scores


def set_deterministic(seed: int, torch_module) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def get_cache_layer(past_key_values, layer_idx: int):
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            return past_key_values.to_legacy_cache()[layer_idx]
        except Exception:
            pass
    try:
        item = past_key_values[layer_idx]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return item[0], item[1]
    except Exception:
        pass
    layers = getattr(past_key_values, "layers", None)
    if layers is not None and len(layers) > layer_idx:
        layer = layers[layer_idx]
        keys = next(
            (
                getattr(layer, name)
                for name in ("keys", "key_cache", "k_cache", "key_states")
                if hasattr(layer, name)
            ),
            None,
        )
        values = next(
            (
                getattr(layer, name)
                for name in ("values", "value_cache", "v_cache", "value_states")
                if hasattr(layer, name)
            ),
            None,
        )
        if keys is not None and values is not None:
            return keys, values
    raise RuntimeError(
        f"Could not read layer {layer_idx} from cache type "
        f"{type(past_key_values).__name__}"
    )


def flatten_key_tensor(tensor, torch_module):
    if tensor.ndim != 4:
        raise RuntimeError(f"Expected key tensor rank 4, got {list(tensor.shape)}")
    return (
        tensor.detach()
        .to("cpu", dtype=torch_module.float32)
        .reshape(-1, tensor.shape[-1])
        .contiguous()
    )


def sample_rows(x, max_rows: int, seed: int, torch_module):
    if x.shape[0] <= max_rows:
        return x
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch_module.randperm(x.shape[0], generator=generator)[:max_rows]
    return x[indices]


def effective_dimension(x, torch_module) -> float:
    if x.shape[0] < 2:
        return 0.0
    centered = x - x.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(x.shape[0] - 1, 1)
    eigenvalues = torch_module.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    denominator = (eigenvalues * eigenvalues).sum()
    if float(total) <= 0 or float(denominator) <= 0:
        return 0.0
    return float((total * total / denominator).item())


def layer_distortion_metrics(
    x,
    *,
    layer: int,
    bits: int,
    max_rows: int,
    ip_pairs: int,
    seed: int,
    compressor_class,
    torch_module,
) -> tuple[float, float, float]:
    x = sample_rows(x, max_rows, seed + layer, torch_module)
    if x.shape[0] < 1:
        raise ValueError("key tensor has no rows")
    dimension = int(x.shape[1])
    compressor = compressor_class(
        head_dim=dimension,
        bits=bits,
        seed=seed,
        device="cpu",
    )
    states = x.reshape(1, 1, x.shape[0], dimension).to(torch_module.float16)
    compressed = compressor.compress(states)
    reconstructed = compressor.decompress(compressed).reshape(x.shape).float()

    # The 2^(2b) factor converts squared b-bit reconstruction error to the
    # bit-normalized constant used by the paper. Per-model min-max ranking is
    # invariant to this common positive scale, but retaining it keeps exported
    # profiling statistics semantically comparable with the formal runs.
    scale = float(2 ** (2 * bits))
    vector_mse = torch_module.mean((x - reconstructed) ** 2, dim=1) * scale
    mse_p95 = float(torch_module.quantile(vector_mse, 0.95).item())

    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(seed + 10007 + layer)
    row_count = int(x.shape[0])
    pair_count = min(ip_pairs, max(1, row_count * max(row_count - 1, 1)))
    left = torch_module.randint(0, row_count, (pair_count,), generator=generator)
    right = torch_module.randint(0, row_count, (pair_count,), generator=generator)
    reference_ip = (x[left] * x[right]).sum(dim=-1)
    reconstructed_ip = (reconstructed[left] * reconstructed[right]).sum(dim=-1)
    ip_error = (
        torch_module.abs(reference_ip - reconstructed_ip)
        * scale
        / math.sqrt(max(dimension, 1))
    )
    ip_p95 = float(torch_module.quantile(ip_error, 0.95).item())
    return mse_p95, ip_p95, effective_dimension(x, torch_module)


def _load_compressor(turboquant_root: Path):
    root = turboquant_root.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"TurboQuant root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("turboquant.compressors_v3")
    return getattr(module, "MSECompressor")


def _load_model(model_path: Path, device: str, load_in_4bit: bool):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Real key-risk profiling requires torch and transformers."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"
        except Exception as exc:
            raise RuntimeError(
                "--load-in-4bit requires bitsandbytes and accelerate."
            ) from exc
    else:
        kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if not load_in_4bit:
        model.to(device)
    model.eval()
    return torch, model, tokenizer


def profile_model_key_risk(
    *,
    model_path: Path,
    model_id: str,
    context_file: Path,
    turboquant_root: Path,
    context_length: int = 4096,
    bits: int = 2,
    max_rows: int = 20000,
    ip_pairs: int = 8192,
    seed: int = 42,
    device: str = "cuda",
    load_in_4bit: bool = False,
) -> pd.DataFrame:
    if device == "cuda":
        try:
            import torch as torch_probe
        except ImportError as exc:
            raise RuntimeError("CUDA profiling requires torch.") from exc
        if not torch_probe.cuda.is_available():
            raise RuntimeError("CUDA is not available for key-risk profiling.")

    torch_module, model, tokenizer = _load_model(model_path, device, load_in_4bit)
    set_deterministic(seed, torch_module)
    compressor_class = _load_compressor(turboquant_root)

    source = context_file.read_text(encoding="utf-8", errors="ignore")
    source_tokens = tokenizer.encode(source, add_special_tokens=False)
    if not source_tokens:
        raise ValueError("profiling context produced no tokens")
    if len(source_tokens) < context_length:
        raise ValueError(
            f"profiling source has {len(source_tokens)} tokens but "
            f"context_length={context_length}; provide a longer source"
        )
    source_tokens = source_tokens[:context_length]
    context = tokenizer.decode(source_tokens, skip_special_tokens=True)
    prompt = apply_chat_template(
        tokenizer,
        [
            {
                "role": "system",
                "content": "You are a careful assistant. Read the context and answer briefly.",
            },
            {
                "role": "user",
                "content": (
                    "Read this literature context excerpt:\n\n"
                    f"{context}\n\nFor this full-layer key-risk sweep, "
                    "reply with one short sentence."
                ),
            },
        ],
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    model_device = next(model.parameters()).device
    encoded = {name: tensor.to(model_device) for name, tensor in encoded.items()}
    with torch_module.inference_mode():
        outputs = model(**encoded, use_cache=True, return_dict=True)

    layer_count = int(model.config.num_hidden_layers)
    rows: list[dict[str, Any]] = []
    for layer in range(layer_count):
        key_tensor, _ = get_cache_layer(outputs.past_key_values, layer)
        flattened = flatten_key_tensor(key_tensor, torch_module)
        mse_p95, ip_p95, effdim = layer_distortion_metrics(
            flattened,
            layer=layer,
            bits=bits,
            max_rows=max_rows,
            ip_pairs=ip_pairs,
            seed=seed,
            compressor_class=compressor_class,
            torch_module=torch_module,
        )
        rows.append(
            {
                "model_id": model_id,
                "layer": layer,
                "kv_type": "key",
                "context_length": context_length,
                "actual_tokens": int(encoded["input_ids"].shape[1]),
                "quant_bits": bits,
                "mse_p95": mse_p95,
                "ip_p95": ip_p95,
                "effective_dim": effdim,
                "c_mse_upper95": mse_p95,
                "c_ip_upper95": ip_p95,
                "effective_dimension": effdim,
                "completed": True,
                "error": "",
            }
        )
    return compute_risk_scores(pd.DataFrame(rows))
