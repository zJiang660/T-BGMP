"""Template for integrating a real model/KV-cache runtime.

Copy this file outside the repository or adapt it in your environment. Do not
commit model paths or credentials.
"""

from tbgmp.kv_cache_wrapper import GenerationResult


class Backend:
    def __init__(self):
        # Cache tokenizer/model runtimes by model_path. Do not reload them for
        # every policy/case combination.
        self.runtimes = {}

    def get_tokenizer(self, model_path):
        # Return the tokenizer from the same cached runtime used by generate().
        raise NotImplementedError

    def generate(
        self,
        *,
        model_path,
        prompt,
        answer,
        policy_name,
        quantization,
        max_new_tokens,
        seed=0,
        add_special_tokens=True,
    ):
        # TODO:
        # 1. Load the user-supplied model from model_path.
        # 2. Apply quantization.key_bits/value_bits/residual_window.
        # 3. Apply protected_key_bits to quantization.protected_layers.
        # 4. Tokenize with the supplied add_special_tokens setting. The full
        #    runner passes False because prompt is already chat-template rendered.
        # 5. Generate deterministically and record memory/runtime metrics.
        # 6. Return the decoded response. The pipeline recomputes answer match.
        raise NotImplementedError("Connect this adapter to a compatible backend")


def create_backend():
    return Backend()
