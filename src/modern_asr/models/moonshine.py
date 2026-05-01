"""Moonshine adapter (Useful Sensors).

Models:
    - moonshine-tiny: 27M params, edge/on-device deployment
    - moonshine-base: ~245M params

References:
    - https://github.com/usefulsensors/moonshine
    - https://huggingface.co/UsefulSensors/moonshine
"""

from __future__ import annotations

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

    ensure_pypi("onnxruntime>=1.17.0")


@register_model("moonshine-tiny")
class MoonshineTiny(ASRModel):
    """Moonshine-Tiny: 27M params, designed for edge / Raspberry Pi deployment."""

    MODEL_CARD = "https://huggingface.co/UsefulSensors/moonshine"
    SUPPORTED_LANGUAGES = {"en", "auto"}
    SUPPORTED_MODES = {"transcribe"}
    REQUIREMENTS = ["onnxruntime"]

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
        return "UsefulSensors/moonshine-tiny"

    @property
    def model_id(self) -> str:
        return "moonshine-tiny"

    def load(self) -> None:
        logger.info("Loading %s", self.model_id)

        _check_deps()
        try:
            import moonshine  # type: ignore[import-untyped]
            self._model = moonshine
        except ImportError:
            # Fallback to direct ONNX if moonshine package not available
            from modern_asr.backends.onnx_backend import ONNXBackend
            self._backend_impl = ONNXBackend(device="cpu")
            # Moonshine exports encoder/decoder as separate ONNX files
            self._backend_impl.load(self._model_path)
            self._model = None
        self._is_loaded = True

    def transcribe(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        logger.info("Transcribing with %s", self.model_id)

        self._ensure_loaded()
        waveform = self._to_waveform(audio)

        if hasattr(self, "_model") and self._model is not None and hasattr(self._model, "transcribe"):
            text = self._model.transcribe(waveform)  # type: ignore[union-attr]
        else:
            # ONNX fallback placeholder
            text = ""

        return ASRResult(
            text=text.strip() if isinstance(text, str) else "",
            segments=[Segment(text=text.strip() if isinstance(text, str) else "")],
            language="en",
            model_id=self.model_id,
        )

    def _to_waveform(self, audio: AudioInput) -> np.ndarray:
        if audio.is_array():
            arr = audio.data
            if isinstance(arr, np.ndarray):
                return arr
        from modern_asr.utils.audio import load_audio
        loaded = load_audio(str(audio.data), target_sr=16000)
        return loaded.data  # type: ignore[return-value]
