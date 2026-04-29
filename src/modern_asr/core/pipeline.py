"""High-level ASR pipeline for end-to-end recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from modern_asr.core.base import ASRModel
from modern_asr.core.config import BackendConfig, ModelConfig, PipelineConfig
from modern_asr.core.registry import create_model, get_model_class
from modern_asr.core.types import ASRResult, AudioInput
from modern_asr.utils.audio import load_audio


class ASRPipeline:
    """Unified ASR pipeline with automatic model loading and audio preprocessing.

    Usage::

        pipe = ASRPipeline("fireredasr-llm")
        result = pipe("path/to/audio.wav")
        print(result.text)
    """

    def __init__(
        self,
        model_id: str | None = None,
        model_config: ModelConfig | None = None,
        backend: BackendConfig | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            model_id: Canonical model identifier (e.g. ``"sensevoice-small"``).
            model_config: Optional per-model configuration override.
            backend: Optional global backend configuration.
            config_path: Path to a YAML pipeline config file.
        """
        self._model: ASRModel | None = None
        self._backend = backend or BackendConfig()
        self._model_config = model_config
        self._config_path = config_path
        self._pipeline_config: PipelineConfig | None = None

        if config_path:
            self._pipeline_config = PipelineConfig.from_yaml(config_path)
            self._backend = self._pipeline_config.default_backend

        if model_id:
            self._init_model(model_id)

    def _init_model(self, model_id: str) -> None:
        if self._pipeline_config and model_id in self._pipeline_config.models:
            cfg = self._pipeline_config.models[model_id]
        else:
            cfg = self._model_config or ModelConfig(model_id=model_id)

        backend = cfg.backend or self._backend
        self._model = create_model(model_id, config=cfg, backend=backend)
        self._model.load()

    def __call__(
        self,
        audio: str | Path | AudioInput,
        task: str = "transcribe",
        language: str | None = None,
        **kwargs: Any,
    ) -> ASRResult:
        """Run ASR on audio.

        Args:
            audio: File path or pre-wrapped `AudioInput`.
            task: Task type (``"transcribe"``, ``"translate"``, ``"diarize"``, etc.).
            language: Target language code (``"zh"``, ``"en"``, ``"auto"``).
            **kwargs: Additional per-call arguments passed to the model.

        Returns:
            Unified `ASRResult`.
        """
        if self._model is None:
            raise RuntimeError("No model has been initialized. Provide a model_id at construction.")

        if isinstance(audio, (str, Path)):
            audio_input = load_audio(str(audio))
        elif isinstance(audio, np.ndarray):
            audio_input = AudioInput(data=audio, sample_rate=16000)
        else:
            audio_input = audio

        if language:
            kwargs["language"] = language

        if task == "transcribe":
            return self._model.transcribe(audio_input, **kwargs)
        elif task == "translate":
            return self._model.translate(audio_input, **kwargs)
        elif task == "diarize":
            return self._model.diarize(audio_input, **kwargs)
        elif task == "emotion":
            return self._model.detect_emotion(audio_input, **kwargs)
        elif task == "event":
            return self._model.detect_events(audio_input, **kwargs)
        else:
            return self._model.transcribe(audio_input, task=task, **kwargs)

    @property
    def model(self) -> ASRModel | None:
        return self._model

    def switch_model(self, model_id: str) -> None:
        """Hot-swap to another registered model."""
        if self._model is not None:
            self._model.unload()
        self._init_model(model_id)

    def unload(self) -> None:
        if self._model is not None:
            self._model.unload()

    def __enter__(self) -> ASRPipeline:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.unload()
