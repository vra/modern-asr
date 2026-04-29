"""Modern ASR: A unified, extensible toolkit for LLM-based Automatic Speech Recognition."""

from modern_asr.core.base import ASRModel
from modern_asr.core.config import ModelConfig, BackendConfig
from modern_asr.core.pipeline import ASRPipeline
from modern_asr.core.registry import register_model, list_models
from modern_asr.core.types import (
    ASRResult,
    AudioInput,
    Language,
    RecognitionMode,
    Segment,
)

# Trigger auto-discovery and registration of all built-in model adapters
import modern_asr.models as _models_module  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "ASRModel",
    "ASRPipeline",
    "ASRResult",
    "AudioInput",
    "BackendConfig",
    "Language",
    "ModelConfig",

    "RecognitionMode",
    "Segment",
    "list_models",
    "register_model",
]
