"""GLM-ASR adapter (Zhipu AI / 智谱AI).

Models:
    - glm-asr-nano-2512: 1.5B params, open-source

References:
    - https://github.com/zai-org/GLM-ASR
    - https://huggingface.co/zai-org/GLM-ASR-Nano-2512
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

from modern_asr.core.audio_llm import AudioLLMModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.registry import register_model
from modern_asr.core.subprocess_mixin import SubprocessIsolatedMixin
from modern_asr.core.types import ASRResult, AudioInput, Segment
from modern_asr.utils.log import get_logger

logger = get_logger(__name__)


def _glm_transformers_supported() -> bool:
    """Return True if the current transformers has glmasr support."""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        return "glmasr" in CONFIG_MAPPING
    except Exception:
        return False


@register_model("glm-asr-nano-2512")
class GLMASRNano2512(SubprocessIsolatedMixin, AudioLLMModel):
    """GLM-ASR-Nano-2512: 1.5B open-source ASR from Zhipu AI."""

    MODEL_CARD = "https://huggingface.co/zai-org/GLM-ASR-Nano-2512"
    SUPPORTED_LANGUAGES = {"zh", "en", "auto", "multi"}
    SUPPORTED_MODES = {"transcribe"}
    REQUIREMENTS = ["torch", "transformers", "sentencepiece"]

    HF_PATH = "zai-org/GLM-ASR-Nano-2512"
    PROCESSOR_CLS = "transformers.AutoProcessor"
    MODEL_CLS = "transformers.AutoModel"
    DEFAULT_MAX_NEW_TOKENS = 256

    # Subprocess isolation: GLM-ASR requires transformers from source (dev)
    SUBPROCESS_VENV = ".venv_glm"
    SUBPROCESS_ENV_VAR = "MODERN_ASR_GLM_VENV"
    SUBPROCESS_CHECK = staticmethod(lambda: not _glm_transformers_supported())

    def __init__(
        self,
        config: ModelConfig,
        backend: BackendConfig | None = None,
    ) -> None:
        super().__init__(config, backend)

    @property
    def model_id(self) -> str:
        return "glm-asr-nano-2512"

    def load(self) -> None:
        """Load with error handling for missing transformers support."""
        logger.info("Loading %s", self.model_id)
        self._try_native_then_subprocess(native_load=super().load)

    def transcribe(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        """Transcribe audio using GLM-ASR's chat-template input format."""
        self._ensure_loaded()

        # Use subprocess backend if active
        if getattr(self, "_subprocess_backend", None) is not None:
            audio_path = self._resolve_audio_path(audio)
            return self._subprocess_transcribe(audio_path, **kwargs)

        import torch

        lang = kwargs.get("language", self.config.language or "auto")

        # Resolve audio to a file path (GLM processor accepts path, not waveform)
        audio_path = self._resolve_audio_path(audio)

        # Build chat-template messages
        prompt_text = self._prompt_for_language(lang)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "url": audio_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_dtype = next(self._model.parameters()).dtype
        model_inputs = {}
        for k, v in inputs.items():
            if v.dtype == torch.float32 or v.dtype == torch.float64:
                model_inputs[k] = v.to(self._model.device, dtype=model_dtype)
            else:
                model_inputs[k] = v.to(self._model.device)

        gen_kwargs = {
            "max_new_tokens": kwargs.get(
                "max_new_tokens",
                self.config.max_new_tokens or self.DEFAULT_MAX_NEW_TOKENS,
            ),
            "do_sample": False,
        }

        with torch.no_grad():
            outputs = self._model.generate(**model_inputs, **gen_kwargs)

        # Slice off the input prompt tokens
        generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
        text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return ASRResult(
            text=text.strip(),
            segments=[Segment(text=text.strip(), language=lang)],
            language=lang,
            model_id=self.model_id,
        )

    def _resolve_audio_path(self, audio: AudioInput) -> str:
        """Return a file path for the audio (save to temp if needed)."""
        if audio.is_file():
            return str(audio.data)
        # Array input — write to temporary WAV file
        arr = audio.data if audio.is_array() else np.array(audio.data)
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        # Use soundfile for writing (core dependency)
        import soundfile as sf

        sf.write(tmp_path, arr, audio.sample_rate)
        return tmp_path

    def _prompt_for_language(self, lang: str | None) -> str:
        """Return the transcription prompt text for the given language."""
        if lang == "zh":
            return "请将这段音频转录为文本"
        if lang == "en":
            return "Please transcribe this audio into text"
        return "Please transcribe this audio into text"
