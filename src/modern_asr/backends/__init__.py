"""Inference backend abstractions and implementations."""

from modern_asr.backends.base import InferenceBackend
from modern_asr.backends.transformers_backend import TransformersBackend

__all__ = ["InferenceBackend", "TransformersBackend"]
