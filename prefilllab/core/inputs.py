"""Synthetic input construction within the model's actual embedding vocabulary."""

from typing import Any

from prefilllab.errors import BenchmarkError


class SyntheticInputGenerator:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def generate(
        self, tokenizer: Any, vocabulary_size: int, batch_size: int, input_length: int, device: str
    ) -> dict[str, Any]:
        import torch

        valid = sorted(
            {
                token
                for token in tokenizer.get_vocab().values()
                if 0 <= token < vocabulary_size and token not in tokenizer.all_special_ids
            }
        )
        if not valid:
            raise BenchmarkError("Tokenizer has no non-special tokens in the embedding vocabulary.")
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        indices = torch.randint(len(valid), (batch_size, input_length), generator=generator)
        tokens = torch.tensor(valid, dtype=torch.long)[indices].to(device)
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}
