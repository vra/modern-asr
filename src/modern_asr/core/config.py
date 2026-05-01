"""Configuration management for Modern ASR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class BackendConfig(BaseModel):
    """Backend runtime configuration."""

    name: Literal["transformers", "vllm", "onnx", "auto"] = "auto"
    device: str = "auto"  # "auto", "cuda", "cuda:0", "cpu", "mps"
    dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32", "int8", "int4"
    batch_size: int = 1
    num_workers: int = 1

    # Backend-specific kwargs
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Per-model configuration."""

    model_id: str
    model_path: str | Path | None = None
    model_revision: str | None = None

    # Recognition behavior
    language: str = "auto"
    task: str = "transcribe"  # "transcribe", "translate", "diarize", ...
    return_timestamps: bool = False
    return_word_timestamps: bool = False
    return_diarization: bool = False
    return_emotion: bool = False
    return_events: bool = False

    # Generation / decoding
    max_new_tokens: int | None = None
    beam_size: int = 1
    temperature: float = 0.0
    prompt: str | None = None

    # Backend override (falls back to global backend if None)
    backend: BackendConfig | None = None

    # Model-specific kwargs
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_paths(self) -> ModelConfig:
        if self.model_path is not None and isinstance(self.model_path, str):
            p = Path(self.model_path)
            if p.exists():
                self.model_path = p.resolve()
        return self


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    default_backend: BackendConfig = Field(default_factory=BackendConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)
