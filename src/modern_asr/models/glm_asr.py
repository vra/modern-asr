"""GLM-ASR adapter (Zhipu AI / 智谱AI).

Models:
    - glm-asr-nano-2512: 1.5B params, open-source
    - glm-asr-2512: cloud version

References:
    - https://github.com/zai-org/GLM-ASR
    - https://huggingface.co/zai-org/GLM-ASR-Nano-2512
"""

from __future__ import annotations

from typing import Any

from modern_asr.core.audio_llm import AudioLLMModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.registry import register_model



from modern_asr.utils.log import get_logger

logger = get_logger(__name__)

@register_model("glm-asr-nano-2512")
class GLMASRNano2512(AudioLLMModel):
    """GLM-ASR-Nano-2512: 1.5B open-source ASR from Zhipu AI."""

    MODEL_CARD = "https://huggingface.co/zai-org/GLM-ASR-Nano-2512"
    SUPPORTED_LANGUAGES = {"zh", "en", "auto", "multi"}
    SUPPORTED_MODES = {"transcribe"}
    REQUIREMENTS = ["torch", "transformers", "sentencepiece"]

    HF_PATH = "zai-org/GLM-ASR-Nano-2512"
    PROCESSOR_CLS = "transformers.AutoTokenizer"
    MODEL_CLS = "transformers.AutoModelForSpeechSeq2Seq"
    DEFAULT_MAX_NEW_TOKENS = 256

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

        try:
            super().load()
        except (ValueError, ImportError) as exc:
            raise RuntimeError(
                f"GLM-ASR requires a newer version of transformers. "
                f"Please upgrade: uv sync --extra transformers. "
                f"Original error: {exc}"
            ) from exc
