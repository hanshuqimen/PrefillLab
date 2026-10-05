"""Errors exposed by the library, CLI, and HTTP boundary."""


class PrefillLabError(Exception):
    """Base class for actionable PrefillLab failures."""


class BackendUnavailableError(PrefillLabError):
    """A backend is unavailable or not implemented."""


class ModelLoadError(PrefillLabError):
    """A model or tokenizer could not be loaded."""


class BenchmarkError(PrefillLabError):
    """Inference failed or the experiment is invalid."""


class ProfilerError(PrefillLabError):
    """A requested profiler failed."""
