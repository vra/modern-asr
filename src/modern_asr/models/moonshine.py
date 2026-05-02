"""Moonshine adapter (Useful Sensors).

Models:
    - moonshine-tiny: 27M params, edge/on-device deployment
    - moonshine-base: ~245M params

References:
    - https://github.com/usefulsensors/moonshine
    - https://huggingface.co/UsefulSensors/moonshine
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from modern_asr.core.base import ASRModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.registry import register_model
from modern_asr.core.types import ASRResult, AudioInput, Segment

from modern_asr.utils.log import get_logger

logger = get_logger(__name__)


def _check_deps() -> None:
    logger.info("Checking dependencies for %s", __name__)

    from modern_asr.utils.auto_install import ensure_pypi

    ensure_pypi("useful-moonshine", "moonshine")


@register_model("moonshine-tiny")
class MoonshineTiny(ASRModel):
    """Moonshine-Tiny: 27M params, designed for edge / Raspberry Pi deployment."""

    MODEL_CARD = "https://huggingface.co/UsefulSensors/moonshine"
    SUPPORTED_LANGUAGES = {"en", "auto"}
    SUPPORTED_MODES = {"transcribe"}
    REQUIREMENTS = ["useful-moonshine"]

    def __init__(
        self,
        config: ModelConfig,
        backend: BackendConfig | None = None,
    ) -> None:
        super().__init__(config, backend)
        self._model_path = self._resolve_model_path()

    def _resolve_model_path(self) -> str:
        if self.config.model_path:
            return str(self.config.model_path)
        return "moonshine/tiny"

    @property
    def model_id(self) -> str:
        return "moonshine-tiny"

    def load(self) -> None:
        logger.info("Loading %s", self.model_id)

        _check_deps()
        os.environ.setdefault("KERAS_BACKEND", "torch")
        import moonshine  # type: ignore[import-untyped]

        self._model = moonshine
        self._is_loaded = True

    def transcribe(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        logger.info("Transcribing with %s", self.model_id)

        self._ensure_loaded()
        audio_path = self._resolve_audio_path(audio)
        texts = self._model.transcribe(audio_path, self._model_path)
        text = texts[0] if isinstance(texts, list) else str(texts)

        return ASRResult(
            text=text.strip() if isinstance(text, str) else "",
            segments=[Segment(text=text.strip() if isinstance(text, str) else "")],
            language="en",
            model_id=self.model_id,
        )

    def _resolve_audio_path(self, audio: AudioInput) -> str:
        if audio.is_file():
            return str(audio.data)
        # Moonshine expects a file path, so write temporary wav
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            arr = audio.data if audio.is_array() else audio.data
            sf.write(f.name, arr, 16000)
            return f.name
