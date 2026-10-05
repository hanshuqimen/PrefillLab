"""Local decoder-only Hugging Face inference with independent prefill and decode."""

import logging
from typing import Any

from prefilllab.core.inputs import SyntheticInputGenerator
from prefilllab.errors import BackendUnavailableError, BenchmarkError, ModelLoadError

logger = logging.getLogger(__name__)


class TransformersBackend:
    simulated = False

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.device = "cpu"
        self.model: Any = None
        self.tokenizer: Any = None
        self.torch: Any = None

    def load_model(self, model: str, dtype: str, device: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except (ImportError, OSError) as exc:
            raise BackendUnavailableError(
                "Transformers backend needs torch and transformers. Install prefilllab[transformers]."
            ) from exc
        self.torch = torch
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise BackendUnavailableError(
                    "CUDA is unavailable. Use --device cpu or install a CUDA PyTorch build."
                )
            torch.cuda.set_device(self.device)
        if self.device == "cpu" and dtype == "float16":
            raise BenchmarkError(
                "float16 CPU kernels are model-dependent; use float32 or bfloat16."
            )
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=getattr(torch, dtype),
                trust_remote_code=False,
            )
            self.model = self.model.to(self.device)
            if self.model.config.is_encoder_decoder:
                raise ModelLoadError("v0.1.0 supports decoder-only causal language models.")
            self.model.eval()
        except ModelLoadError:
            raise
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            raise ModelLoadError(f"Could not load '{model}': {exc}") from exc

    def prepare_inputs(self, batch_size: int, input_length: int) -> dict[str, Any]:
        maximum = getattr(self.model.config, "max_position_embeddings", None)
        if maximum is not None and input_length > maximum:
            raise BenchmarkError(
                f"Input length {input_length} exceeds the model context limit {maximum}."
            )
        vocabulary_size = self.model.get_input_embeddings().num_embeddings
        return SyntheticInputGenerator(self.seed).generate(
            self.tokenizer,
            vocabulary_size,
            batch_size,
            input_length,
            self.device,
        )

    def prefill(self, inputs: Any) -> Any:
        with self.torch.inference_mode():
            return self.model(**inputs, use_cache=True, return_dict=True)

    def first_token(self, output: Any) -> Any:
        with self.torch.inference_mode():
            return output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    def decode(self, inputs: Any, output: Any, token: Any, max_new_tokens: int) -> Any:
        if max_new_tokens <= 1:
            return token
        if output.past_key_values is None:
            raise BenchmarkError(
                "Model did not return a KV cache; cached decode cannot be measured."
            )
        maximum = getattr(self.model.config, "max_position_embeddings", None)
        if maximum is not None and inputs["input_ids"].shape[1] + max_new_tokens - 1 > maximum:
            raise BenchmarkError(
                "Input plus requested decode tokens exceeds the model context limit."
            )
        generated = [token]
        cache, mask = output.past_key_values, inputs["attention_mask"]
        with self.torch.inference_mode():
            for _ in range(max_new_tokens - 1):
                mask = self.torch.cat((mask, self.torch.ones_like(token)), dim=1)
                output = self.model(
                    input_ids=token,
                    attention_mask=mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                token = self.first_token(output)
                cache = output.past_key_values
                generated.append(token)
            return self.torch.cat(generated, dim=1)

    def generate(self, inputs: Any, max_new_tokens: int) -> Any:
        output = self.prefill(inputs)
        return self.decode(inputs, output, self.first_token(output), max_new_tokens)

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def cleanup(self) -> None:
        self.model = None
        self.tokenizer = None
        if self.torch is not None and self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
