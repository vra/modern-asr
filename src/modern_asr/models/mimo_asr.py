"""MiMo-V2.5-ASR adapter (Xiaomi / 小米).

Models:
    - mimo-asr-v2.5: End-to-end ASR, supports dialects, code-switching, noise,
      multi-speaker, song lyrics, and knowledge-intensive content.

Architecture
------------
MiMo-V2.5-ASR is an **Audio-LLM** model with two components:

1. **MiMo-Audio-Tokenizer** — encodes raw audio into discrete tokens.
2. **MiMo-V2.5-ASR** (8B causal LM) — decodes audio tokens + optional
   language tags into text.

Because the inference code lives in the official GitHub repo (not on
PyPI or inside ``transformers``), this adapter locates a local clone,
injects ``src/`` into ``sys.path``, and wraps the ``MimoAudio`` class.

Prerequisites
-------------
* Python >= 3.12 (upstream tests on 3.12)
* CUDA >= 12.0
* ``flash-attn >= 2.7.4`` (strongly recommended for speed)
* Local clone of the official repo + downloaded weights

Quick-start::

    git clone https://github.com/XiaomiMiMo/MiMo-V2.5-ASR.git
    cd MiMo-V2.5-ASR
    pip install -r requirements.txt

    huggingface-cli download XiaomiMiMo/MiMo-V2.5-ASR \
        --local-dir models/MiMo-V2.5-ASR
    huggingface-cli download XiaomiMiMo/MiMo-Audio-Tokenizer \
        --local-dir models/MiMo-Audio-Tokenizer

    # In Python
    from modern_asr import ASRPipeline
    pipe = ASRPipeline("mimo-asr-v2.5")
    result = pipe("audio.wav", language="zh")

License: Apache-2.0
References:
    - https://github.com/XiaomiMiMo/MiMo-V2.5-ASR
    - https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR
    - https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer
"""

from __future__ import annotations

import os
import sys
from typing import Any

from modern_asr.core.audio_llm import AudioLLMModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.registry import register_model
from modern_asr.core.types import ASRResult


def _check_deps() -> None:
    from modern_asr.utils.auto_install import ensure_pypi

    ensure_pypi("torch>=2.0")
    ensure_pypi("transformers>=4.40.0")


@register_model("mimo-asr-v2.5")
class MiMoASRV25(AudioLLMModel):
    """MiMo-V2.5-ASR: Xiaomi's open-source ASR for complex real-world scenarios."""

    MODEL_CARD = "https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR"
    SUPPORTED_LANGUAGES = {"zh", "en", "yue", "wuu", "nan", "cmn", "auto", "multi"}
    SUPPORTED_MODES = {"transcribe", "diarize"}
    REQUIREMENTS = ["torch", "transformers"]

    # Language tags used by MiMo's asr_sft() method
    _LANGUAGE_TAGS = {
        "zh": "<chinese>",
        "en": "<english>",
    }

    _REPO_URL = "https://github.com/XiaomiMiMo/MiMo-V2.5-ASR.git"

    def __init__(
        self,
        config: ModelConfig,
        backend: BackendConfig | None = None,
    ) -> None:
        # We do NOT call AudioLLMModel.__init__ path resolution because MiMo
        # uses a local directory layout rather than a pure HF hub ID.
        super().__init__(config, backend)
        self._model_dir = self._resolve_model_dir()
        self._tokenizer_dir = self._resolve_tokenizer_dir()

    # ------------------------------------------------------------------ #
    # Path resolution
    # ------------------------------------------------------------------ #

    def _resolve_model_dir(self) -> str:
        if self.config.model_path:
            return str(self.config.model_path)
        candidates = [
            "./models/MiMo-V2.5-ASR",
            "./MiMo-V2.5-ASR/models/MiMo-V2.5-ASR",
            "/tmp/MiMo-V2.5-ASR/models/MiMo-V2.5-ASR",
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        return candidates[0]

    def _resolve_tokenizer_dir(self) -> str:
        model = self._model_dir
        candidates = [
            os.path.join(os.path.dirname(model), "MiMo-Audio-Tokenizer"),
            "./models/MiMo-Audio-Tokenizer",
            "/tmp/MiMo-V2.5-ASR/models/MiMo-Audio-Tokenizer",
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        return candidates[0]

    def _find_repo_root(self) -> str | None:
        """Walk upward from model_dir to find the repo root containing src/."""
        path = os.path.abspath(self._model_dir)
        for _ in range(5):
            if os.path.isdir(os.path.join(path, "src", "mimo_audio")):
                return path
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
        return None

    @property
    def model_id(self) -> str:
        return "mimo-asr-v2.5"

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        from modern_asr.utils.log import get_logger

        logger = get_logger(__name__)
        logger.info("Loading %s", self.model_id)

        _check_deps()

        from modern_asr.utils.auto_install import ensure_git, ensure_hf

        # 1. Auto-clone the official repo if missing
        repo_root = ensure_git(self._REPO_URL, "MiMo-V2.5-ASR")
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        # 2. Auto-download HF weights if missing
        model_dir = ensure_hf("XiaomiMiMo/MiMo-V2.5-ASR", "mimo-v2.5-asr")
        tokenizer_dir = ensure_hf(
            "XiaomiMiMo/MiMo-Audio-Tokenizer", "mimo-audio-tokenizer"
        )

        from src.mimo_audio.mimo_audio import MimoAudio

        self._model = MimoAudio(str(model_dir), str(tokenizer_dir))
        self._is_loaded = True
        logger.info("%s ready", self.model_id)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def transcribe(self, audio: Any, **kwargs: Any) -> ASRResult:
        self._ensure_loaded()
        audio_path = self._audio_to_file(audio)

        lang = kwargs.get("language", self.config.language)
        audio_tag = self._LANGUAGE_TAGS.get(lang or "", "")

        text = self._model.asr_sft(audio_path, audio_tag=audio_tag)
        return self._build_result(text, **kwargs)

    def diarize(self, audio: Any, **kwargs: Any) -> ASRResult:
        """MiMo-V2.5-ASR supports multi-speaker transcription via ASR."""
        return self.transcribe(audio, **kwargs)
