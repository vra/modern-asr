"""OpenAI Whisper adapter.

Models:
    - whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large-v3
    - whisper-large-v3-turbo: faster variant

References:
    - https://github.com/openai/whisper
    - https://huggingface.co/openai
"""

from __future__ import annotations

from typing import Any

import numpy as np

from modern_asr.core.base import ASRModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.registry import register_model
from modern_asr.core.types import ASRResult, AudioInput, Segment, WordTimestamp


def _check_deps() -> None:
    try:
        import whisper  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Whisper requires 'openai-whisper' and 'torch'. "
            "Install with: uv pip install modern-asr[whisper]"
        ) from exc


@register_model("whisper-large-v3")
class WhisperLargeV3(ASRModel):
    """OpenAI Whisper Large V3: 99+ languages, robust general-purpose ASR."""

    MODEL_CARD = "https://huggingface.co/openai/whisper-large-v3"
    SUPPORTED_LANGUAGES = {
        "zh", "en", "yue", "ja", "ko", "fr", "de", "es", "ru", "auto", "multi"
    }
    SUPPORTED_MODES = {"transcribe", "translate"}
    REQUIREMENTS = ["openai-whisper", "torch"]

    def __init__(
        self,
        config: ModelConfig,
        backend: BackendConfig | None = None,
    ) -> None:
        super().__init__(config, backend)
        self._model_name = self._resolve_model_name()

    def _resolve_model_name(self) -> str:
        if self.config.model_path:
            return str(self.config.model_path)
        size_map = {
            "whisper-tiny": "tiny",
            "whisper-base": "base",
            "whisper-small": "small",
            "whisper-medium": "medium",
            "whisper-large-v3": "large-v3",
            "whisper-large-v3-turbo": "large-v3-turbo",
        }
        return size_map.get(self.config.model_id, "large-v3")

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def load(self) -> None:
        _check_deps()
        import whisper

        self._model = whisper.load_model(self._model_name)
        self._is_loaded = True

    def transcribe(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        self._ensure_loaded()
        waveform = self._to_waveform(audio)

        options = {
            "language": kwargs.get("language", self.config.language or None),
            "task": kwargs.get("task", "transcribe"),
            "temperature": kwargs.get("temperature", self.config.temperature or 0.0),
            "word_timestamps": kwargs.get(
                "word_timestamps", self.config.return_word_timestamps
            ),
        }
        result = self._model.transcribe(waveform, **options)  # type: ignore[union-attr]

        text = result.get("text", "").strip()
        segments = []
        for seg in result.get("segments", []):
            words = []
            for w in seg.get("words", []):
                words.append(
                    WordTimestamp(
                        text=w.get("word", ""),
                        start=w.get("start", 0.0),
                        end=w.get("end", 0.0),
                    )
                )
            segments.append(
                Segment(
                    text=seg.get("text", ""),
                    start=seg.get("start"),
                    end=seg.get("end"),
                    words=words,
                )
            )

        return ASRResult(
            text=text,
            segments=segments,
            language=result.get("language"),
            model_id=self.model_id,
            extra={"raw": result},
        )

    def translate(self, audio: AudioInput, target_language: str = "en", **kwargs: Any) -> ASRResult:
        return self.transcribe(audio, task="translate", language=target_language, **kwargs)

    def _to_waveform(self, audio: AudioInput) -> np.ndarray:
        if audio.is_array():
            arr = audio.data
            if isinstance(arr, np.ndarray):
                return arr
        from modern_asr.utils.audio import load_audio
        loaded = load_audio(str(audio.data), target_sr=16000)
        return loaded.data  # type: ignore[return-value]


# Register common Whisper size aliases
for _alias, _size in {
    "whisper-tiny": "tiny",
    "whisper-base": "base",
    "whisper-small": "small",
    "whisper-medium": "medium",
    "whisper-large-v3-turbo": "large-v3-turbo",
}.items():
    # Dynamically create lightweight proxy classes for each size
    _cls = type(
        f"Whisper{_size.replace('-', '_').title()}",
        (WhisperLargeV3,),
        {"_size": _size},
    )
    register_model(_alias)(_cls)
