"""Qwen3-ASR adapter (Alibaba Qwen Team / 阿里云).

Models:
    - qwen3-asr-0.6b: 600M params, RTF 0.009
    - qwen3-asr-1.7b: 1.7B params, RTF 0.015
    - Supports 22 Chinese dialects, 52 languages total

Installation::

    pip install git+https://github.com/QwenLM/Qwen3-ASR.git

References:
    - https://github.com/QwenLM/Qwen3-ASR
"""

from __future__ import annotations

from typing import Any

from modern_asr.core.audio_llm import AudioLLMModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.registry import register_model
from modern_asr.core.types import ASRResult



from modern_asr.utils.log import get_logger

logger = get_logger(__name__)

def _check_deps() -> None:
    logger.info("Checking dependencies for %s", __name__)

    from modern_asr.utils.auto_install import ensure_pypi

    ensure_pypi("torch>=2.0")
    ensure_pypi("qwen-asr>=0.0.6", "qwen_asr")


class _QwenASRBase(AudioLLMModel):
    """Shared logic for Qwen3-ASR variants."""

    SUPPORTED_LANGUAGES = {
        "zh", "en", "yue", "ja", "ko", "auto", "multi",
        "wuu", "cmn", "nan", "hak", "gan", "hsn", "czh", "cjy",
    }
    SUPPORTED_MODES = {"transcribe", "translate"}
    REQUIREMENTS = ["torch", "qwen-asr"]

    # Qwen3-ASR uses full language names rather than ISO codes
    _LANGUAGE_MAP = {
        "zh": "Chinese",
        "en": "English",
        "yue": "Cantonese",
        "ja": "Japanese",
        "ko": "Korean",
        "auto": None,
        "multi": None,
    }

    def __init__(
        self,
        config: ModelConfig,
        backend: BackendConfig | None = None,
    ) -> None:
        super().__init__(config, backend)
        self._hf_path = self._resolve_hf_path()

    def _resolve_hf_path(self) -> str:
        if self.config.model_path:
            return str(self.config.model_path)
        defaults = {
            "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
            "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
        }
        return defaults.get(self.config.model_id, f"Qwen/{self.config.model_id}")

    def load(self) -> None:
        logger.info("Loading %s", self.model_id)

        _check_deps()
        from qwen_asr import Qwen3ASRModel

        device = self._resolve_device()
        dtype = self._resolve_dtype()

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if device != "cpu":
            kwargs["device_map"] = device
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        self._model = Qwen3ASRModel.from_pretrained(
            self._hf_path,
            max_new_tokens=self.config.max_new_tokens or 512,
            **kwargs,
        )
        self._is_loaded = True

    def transcribe(self, audio: Any, **kwargs: Any) -> ASRResult:
        logger.info("Transcribing with %s", self.model_id)

        self._ensure_loaded()

        lang = kwargs.get("language", self.config.language)
        qwen_lang = self._LANGUAGE_MAP.get(lang, lang)

        audio_path = self._audio_to_file(audio)

        results = self._model.transcribe(
            audio_path,
            language=qwen_lang,
            return_time_stamps=False,
        )

        text = results[0].text.strip() if results else ""
        detected_lang = results[0].language if results else qwen_lang or "auto"

        kw = dict(kwargs)
        kw.pop("language", None)
        return self._build_result(text, language=detected_lang, **kw)


@register_model("qwen3-asr-0.6b")
class Qwen3ASR06B(_QwenASRBase):
    """Qwen3-ASR-0.6B: 600M params, 52 languages, 22 Chinese dialects."""

    MODEL_CARD = "https://huggingface.co/Qwen/Qwen3-ASR-0.6B"

    @property
    def model_id(self) -> str:
        return "qwen3-asr-0.6b"


@register_model("qwen3-asr-1.7b")
class Qwen3ASR17B(_QwenASRBase):
    """Qwen3-ASR-1.7B: 1.7B params, higher accuracy."""

    MODEL_CARD = "https://huggingface.co/Qwen/Qwen3-ASR-1.7B"

    @property
    def model_id(self) -> str:
        return "qwen3-asr-1.7b"
