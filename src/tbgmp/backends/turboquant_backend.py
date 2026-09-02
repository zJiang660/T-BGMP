from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

from .base import GenerationRequest, GenerationResult
from ..turboquant_patch_validation import validate_runtime_contract


INTEGRATION_MESSAGE = (
    "Full model execution requires the external TurboQuant backend. "
    "See docs/backend_integration.md."
)


class TurboQuantBackend:
    """Validated integration boundary for the external TurboQuant runtime.

    The upstream runtime currently exposes TurboQuantV3 and example cache
    wrappers, but its protected-layer option protects first/last layer counts.
    T-BGMP requires arbitrary ranked key-layer IDs and key-only protection.
    This adapter therefore refuses to approximate the method silently.
    """

    request_api = True

    def __init__(
        self,
        turboquant_root: str | None = None,
        device: str = "cuda",
        **_: Any,
    ):
        configured = turboquant_root or os.environ.get("TURBOQUANT_ROOT")
        self.root = Path(configured).expanduser().resolve() if configured else None
        self.device = device
        self.runtime_module = None
        self.compressor_class = None
        self.cache_class = None
        self._import_error = ""
        self._contract_status: dict[str, Any] = {
            "signature_ok": False,
            "behavior_ok": False,
            "passed": False,
        }
        self._runtime_cache: dict[str, dict[str, Any]] = {}
        self.model_load_count = 0
        self._compressors_module = None
        self._compressors_file = (
            self.root / "turboquant" / "compressors_v3.py" if self.root else None
        )
        self._generation_file = (
            self.root / "turboquant" / "generation_test.py" if self.root else None
        )

        if self.root and self.root.is_dir():
            root_text = str(self.root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            try:
                self.runtime_module = importlib.import_module("turboquant")
                self._compressors_module = importlib.import_module(
                    "turboquant.compressors_v3"
                )
                if hasattr(self._compressors_module, "TurboQuantV3"):
                    self.compressor_class = self._compressors_module.TurboQuantV3
                generation_module = importlib.import_module(
                    "turboquant.generation_test"
                )
                self.cache_class = getattr(generation_module, "V3Cache", None)
                if self.compressor_class is not None and self.cache_class is not None:
                    self._contract_status = validate_runtime_contract(
                        self.compressor_class,
                        self.cache_class,
                    )
            except Exception as exc:  # pragma: no cover - message asserted indirectly
                self._import_error = repr(exc)

    def check_available(self) -> dict[str, Any]:
        root_ok = self.root is not None and self.root.is_dir()
        import_ok = self.runtime_module is not None and not self._import_error
        compressor_ok = self.compressor_class is not None
        compressors_file_ok = (
            self._compressors_file is not None and self._compressors_file.is_file()
        )
        generation_file_ok = (
            self._generation_file is not None and self._generation_file.is_file()
        )
        arbitrary_patch_detected = bool(self._contract_status.get("passed"))
        ready = bool(
            root_ok
            and import_ok
            and compressor_ok
            and generation_file_ok
            and arbitrary_patch_detected
        )
        return {
            "backend": "turboquant",
            "root_configured": self.root is not None,
            "root_exists": root_ok,
            "compressors_v3_file": compressors_file_ok,
            "generation_test_file": generation_file_ok,
            "import_ok": import_ok,
            "compressor_v3_found": compressor_ok,
            "protected_layers_semantics": (
                "explicit_key_layer_ids"
                if arbitrary_patch_detected
                else "prefix_suffix_count"
            ),
            "arbitrary_patch_detected": arbitrary_patch_detected,
            "arbitrary_protected_key_layer_ids": arbitrary_patch_detected,
            "key_only_protection": arbitrary_patch_detected,
            "patch_signature_ok": bool(self._contract_status.get("signature_ok")),
            "patch_behavior_ok": bool(self._contract_status.get("behavior_ok")),
            "patch_contract": self._contract_status,
            "residual_window": "supported by upstream V3 cache",
            "ready_for_tbgmp_generation": ready,
            "error": self._import_error,
            "message": (
                "The external TurboQuant runtime can be inspected. Real "
                "generation is enabled only after API-signature and key-only "
                "behavior checks pass and required Python packages are available."
            ),
        }

    def _raise_if_unavailable(self) -> None:
        status = self.check_available()
        if not status["root_configured"]:
            raise RuntimeError(f"{INTEGRATION_MESSAGE} Set TURBOQUANT_ROOT first.")
        if not status["root_exists"]:
            raise RuntimeError(
                f"{INTEGRATION_MESSAGE} The configured TurboQuant directory "
                "does not exist."
            )
        if not status["import_ok"]:
            raise RuntimeError(
                f"{INTEGRATION_MESSAGE} The external package could not be imported."
            )
        if not status["compressor_v3_found"]:
            raise RuntimeError(
                f"{INTEGRATION_MESSAGE} TurboQuantV3 was not found in the "
                "configured runtime."
            )
        if not status["arbitrary_patch_detected"]:
            raise RuntimeError(
                "Full TurboQuant generation is unavailable because the public "
                "TurboQuant runtime does not directly expose arbitrary "
                "risk-ranked protected key-layer IDs. Apply the patch described "
                "in docs/turboquant_patch_guide.md or provide a compatible "
                "backend."
            )

    def _build_cache(self, request: GenerationRequest, n_layers: int):
        policy = request.policy
        policy_name = str(policy.get("name", "")).lower()
        if policy_name == "fp16" or int(policy.get("key_bits", 16)) >= 16:
            return None

        generation = importlib.import_module("turboquant.generation_test")
        cache_class = getattr(generation, "V3Cache")
        protected_ids = policy.get("protected_layer_ids", policy.get("protected_layers", []))
        if isinstance(protected_ids, str):
            protected_ids = [
                int(value.strip())
                for value in protected_ids.split(",")
                if value.strip()
            ]
        return cache_class(
            key_bits=int(policy.get("key_bits", policy.get("default_key_bits", 4))),
            value_bits=int(
                policy.get("value_bits", policy.get("default_value_bits", 2))
            ),
            residual_window=int(policy.get("residual_window", 128)),
            protected_layers=int(policy.get("protected_layers_count", 0)),
            protected_layer_ids=list(protected_ids),
            protected_key_bits=int(policy.get("protected_key_bits", 8)),
            n_layers=n_layers,
        )

    def _load_runtime(self, model_path: str) -> dict[str, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Real TurboQuant generation requires torch and transformers "
                "in the active environment."
            ) from exc

        path = Path(model_path)
        if not model_path or not path.is_dir():
            raise RuntimeError("Model path does not exist or was not provided.")
        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if device == "cuda" else torch.float32
        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                dtype=dtype,
                **model_kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=dtype,
                **model_kwargs,
            )
        model.to(device)
        model.eval()
        self.model_load_count += 1
        return {
            "torch": torch,
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
        }

    def _get_runtime(self, model_path: str) -> dict[str, Any]:
        key = str(Path(model_path).expanduser().resolve())
        if key not in self._runtime_cache:
            self._runtime_cache[key] = self._load_runtime(model_path)
        return self._runtime_cache[key]

    def get_tokenizer(self, model_path: str):
        """Return the tokenizer owned by the backend's single model runtime."""
        self._raise_if_unavailable()
        return self._get_runtime(model_path)["tokenizer"]

    def generate(
        self,
        request: GenerationRequest | None = None,
        **legacy_kwargs: Any,
    ) -> GenerationResult:
        if request is None:
            quantization = legacy_kwargs.get("quantization")
            policy = {
                "name": legacy_kwargs.get("policy_name", ""),
                "key_bits": getattr(quantization, "key_bits", 16),
                "value_bits": getattr(quantization, "value_bits", 16),
                "protected_key_bits": getattr(
                    quantization, "protected_key_bits", None
                ),
                "protected_layers": list(
                    getattr(quantization, "protected_layers", ())
                ),
                "residual_window": getattr(quantization, "residual_window", 0),
            }
            request = GenerationRequest(
                model_path=str(legacy_kwargs.get("model_path", "")),
                prompt=str(legacy_kwargs.get("prompt", "")),
                answer=str(legacy_kwargs.get("answer", "")),
                policy=policy,
                max_new_tokens=int(legacy_kwargs.get("max_new_tokens", 32)),
                seed=int(legacy_kwargs.get("seed", 0)),
                add_special_tokens=bool(
                    legacy_kwargs.get("add_special_tokens", True)
                ),
            )

        self._raise_if_unavailable()
        start = time.time()
        runtime = self._get_runtime(request.model_path)
        torch = runtime["torch"]
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        device = runtime["device"]
        torch.manual_seed(request.seed)

        encoded = tokenizer(
            request.prompt,
            return_tensors="pt",
            add_special_tokens=request.add_special_tokens,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        cache = self._build_cache(request, int(model.config.num_hidden_layers))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                past_key_values=cache,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][encoded["input_ids"].shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        runtime_s = time.time() - start
        peak_gpu_gb = None
        if torch.cuda.is_available():
            peak_gpu_gb = torch.cuda.max_memory_allocated() / (1024**3)
        return GenerationResult(
            response=response,
            found=str(request.answer).strip() in response,
            status="success",
            metadata={
                "runtime_s": runtime_s,
                "tok_per_s": (
                    float(len(new_tokens)) / runtime_s if runtime_s > 0 else None
                ),
                "peak_gpu_gb": peak_gpu_gb,
                "generated_tokens": int(len(new_tokens)),
                "actual_context_tokens": int(encoded["input_ids"].shape[1]),
            },
        )


def create_backend(**kwargs: Any) -> TurboQuantBackend:
    return TurboQuantBackend(**kwargs)
