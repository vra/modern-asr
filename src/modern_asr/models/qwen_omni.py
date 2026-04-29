"""Qwen2.5-Omni adapter (Alibaba Qwen Team).

Models:
    - qwen2.5-omni-7b: Fully multimodal, speech understanding + generation
    - qwen2.5-omni-3b: smaller variant

Capabilities:
    - ASR (automatic speech recognition)
    - Speech-to-text conversation
    - Audio understanding and reasoning

References:
    - https://huggingface.co/Qwen/Qwen2.5-Omni-7B
    - https://github.com/QwenLM/Qwen2.5-Omni
"""

from __future__ import annotations

from typing import Any

from modern_asr.core.audio_llm import AudioLLMModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.registry import register_model
from modern_asr.core.types import ASRResult


@register_model("qwen2.5-omni-7b")
class Qwen25Omni7B(AudioLLMModel):
    """Qwen2.5-Omni-7B: Fully multimodal model with strong ASR capabilities."""

    MODEL_CARD = "https://huggingface.co/Qwen/Qwen2.5-Omni-7B"
    SUPPORTED_LANGUAGES = {"zh", "en", "yue", "ja", "ko", "auto", "multi"}
    SUPPORTED_MODES = {"transcribe", "translate", "multi_task"}
    REQUIREMENTS = ["torch", "transformers"]

    HF_PATH = "Qwen/Qwen2.5-Omni-7B"
    PROCESSOR_CLS = "transformers.Qwen2_5OmniProcessor"
    MODEL_CLS = "transformers.Qwen2_5OmniModel"
    DEFAULT_MAX_NEW_TOKENS = 256

    def __init__(
        self,
        config: ModelConfig,
        backend: BackendConfig | None = None,
    ) -> None:
        super().__init__(config, backend)

    @property
    def model_id(self) -> str:
        return "qwen2.5-omni-7b"

    def load(self) -> None:
        """Load with descriptive error if transformers is too old."""
        try:
            super().load()
        except ImportError as exc:
            raise RuntimeError(
                f"Qwen2.5-Omni requires a newer version of transformers. "
                f"Please upgrade: uv sync --extra transformers. "
                f"Original error: {exc}"
            ) from exc

    def _build_inputs(self, audio: Any, **kwargs: Any) -> dict[str, Any]:
        """Qwen2.5-Omni expects a list of audios and a text prompt."""
        waveform = self._to_waveform(audio)
        return self._processor(
            audios=[waveform],
            sampling_rate=audio.sample_rate,
            text="Transcribe the speech into text.",
            return_tensors="pt",
        )

    def transcribe(self, audio: Any, **kwargs: Any) -> ASRResult:
        """Override to handle Qwen2.5-Omni's special input format."""
        self._ensure_loaded()
        return self._transcribe_single(audio, **kwargs)
