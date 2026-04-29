"""Core abstractions for Modern ASR."""

from modern_asr.core.audio_llm import AudioLLMModel
from modern_asr.core.base import ASRModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.pipeline import ASRPipeline
from modern_asr.core.registry import list_models, register_model
from modern_asr.core.types import (
    ASRResult,
    AudioInput,
    Language,
    RecognitionMode,
    Segment,
    WordTimestamp,
)

__all__ = [
    "ASRModel",
    "AudioLLMModel",
    "ASRPipeline",
    "ASRResult",
    "AudioInput",
    "BackendConfig",
    "Language",
    "ModelConfig",
    "ModelRegistry",
    "RecognitionMode",
    "Segment",
    "WordTimestamp",
    "list_models",
    "register_model",
]
