"""NVIDIA Canary-Qwen adapter.

Models:
    - canary-qwen-2.5b: FastConformer encoder + Qwen3-1.7B decoder, SALM architecture
    - canary-1b: multilingual (en/de/fr/es)
    - canary-1b-flash: optimized for speed, 1000+ RTFx

References:
    - https://huggingface.co/nvidia/canary-1b
    - https://huggingface.co/nvidia/canary-1b-flash
    - NeMo toolkit: https://github.com/NVIDIA/NeMo
"""

from __future__ import annotations

from typing import Any

import numpy as np

from modern_asr.core.base import ASRModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.registry import register_model
from modern_asr.core.types import ASRResult, AudioInput, Segment


def _check_deps() -> None:
    from modern_asr.utils.auto_install import ensure_pypi

    ensure_pypi("nemo-toolkit[asr]", "nemo")


@register_model("canary-qwen-2.5b")
class CanaryQwen25B(ASRModel):
    """Canary-Qwen 2.5B: #1 on HuggingFace Open ASR Leaderboard, 5.63% WER average."""

    MODEL_CARD = "https://huggingface.co/nvidia/canary-1b"
    SUPPORTED_LANGUAGES = {"en", "de", "fr", "es", "auto", "multi"}
    SUPPORTED_MODES = {"transcribe", "translate"}
    REQUIREMENTS = ["nemo-toolkit[asr]", "torch"]

    def __init__(
        self,
        config: ModelConfig,
        backend: BackendConfig | None = None,
    ) -> None:
        super().__init__(config, backend)
        self._nemo_path = self._resolve_nemo_path()

    def _resolve_nemo_path(self) -> str:
        if self.config.model_path:
            return str(self.config.model_path)
        # NeMo uses model names like "nvidia/canary-1b"
        return "nvidia/canary-1b"

    @property
    def model_id(self) -> str:
        return "canary-qwen-2.5b"

    def load(self) -> None:
        _check_deps()
        from nemo.collections.asr.models import EncDecMultiTaskModel

        self._model = EncDecMultiTaskModel.from_pretrained(self._nemo_path)
        self._model.eval()
        self._is_loaded = True

    def transcribe(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        self._ensure_loaded()
        waveform = self._to_waveform(audio)

        # NeMo expects list of numpy arrays or file paths
        if isinstance(waveform, np.ndarray):
            inputs = [waveform]
        else:
            inputs = [str(audio.data)]

        predictions = self._model.transcribe(  # type: ignore[union-attr]
            inputs,
            batch_size=kwargs.get("batch_size", 1),
            pnc=True,  # punctuation and capitalization
        )

        text = predictions[0].text if hasattr(predictions[0], "text") else str(predictions[0])
        return ASRResult(
            text=text.strip(),
            segments=[Segment(text=text.strip())],
            language=kwargs.get("language", self.config.language),
            model_id=self.model_id,
        )

    def _to_waveform(self, audio: AudioInput) -> np.ndarray:
        if audio.is_array():
            arr = audio.data
            if isinstance(arr, np.ndarray):
                return arr
        from modern_asr.utils.audio import load_audio
        loaded = load_audio(str(audio.data))
        return loaded.data  # type: ignore[return-value]
