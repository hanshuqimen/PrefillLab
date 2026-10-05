"""Real CPU integration tests using local random tiny weights, never a network model."""

import pytest

from prefilllab import Benchmark
from prefilllab.backends.transformers import TransformersBackend
from prefilllab.errors import BenchmarkError

pytestmark = pytest.mark.integration


@pytest.fixture
def tiny_model(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.manual_seed(42)
    torch.set_num_threads(1)
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[EOS]": 1, **{f"token{i}": i for i in range(2, 64)}}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]")
    fast.save_pretrained(tmp_path)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_positions=64,
            n_embd=32,
            n_layer=2,
            n_head=2,
            bos_token_id=1,
            eos_token_id=1,
        )
    )
    model.save_pretrained(tmp_path)
    return str(tmp_path)


def test_real_prefill_decode_profiles(tiny_model):
    result = Benchmark(model=tiny_model, backend="transformers", device="cpu").run(
        input_length=16, output_length=4, warmup_runs=1, benchmark_runs=3
    )
    assert not result.simulated
    assert len(result.samples) == 3
    assert result.ttft_ms >= result.metrics.prefill_latency_ms > 0
    assert result.metrics.decode_latency_ms > 0
    assert result.metrics.peak_gpu_memory_mb is None
    assert result.metrics.gpu_utilization is None
    assert result.breakdown.source == "cpu_module_hooks_exclusive"
    assert result.breakdown.qkv_time_ms > 0
    assert result.layers and result.kernels
    assert all(layer.time_ms >= 0 for layer in result.layers)
    assert all(kernel.device_time_ms is None for kernel in result.kernels)
    assert all("profiling failed" not in warning.lower() for warning in result.warnings)


def test_synthetic_tokens_and_cached_decode(tiny_model):
    backend = TransformersBackend()
    try:
        backend.load_model(tiny_model, "float32", "cpu")
        inputs = backend.prepare_inputs(2, 12)
        assert tuple(inputs["input_ids"].shape) == (2, 12)
        assert inputs["input_ids"].min() >= 2
        assert inputs["input_ids"].max() < 64
        assert backend.generate(inputs, 5).shape == (2, 5)
        with pytest.raises(BenchmarkError):
            backend.prepare_inputs(1, 65)
    finally:
        backend.cleanup()


def test_disable_profile(tiny_model):
    result = Benchmark(model=tiny_model, backend="transformers", profile=False).run(
        input_length=8, warmup_runs=0, benchmark_runs=1
    )
    assert result.breakdown.source == "unavailable"
    assert result.layers == []
