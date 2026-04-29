"""Fun-ASR adapter (Alibaba Tongyi Lab / 阿里通义实验室).

Models:
    - funasr-nano: 0.8B params, tens of millions of hours training data
    - funasr-large: 7.7B params (proprietary API, listed for reference)
    - paraformer-zh: 220M, mature streaming model
    - paraformer-large: large variant

Capabilities:
    - Streaming and non-streaming ASR
    - 7 dialects, 26 accents
    - VAD + ASR + Punctuation pipeline

References:
    - https://github.com/FunAudioLLM/Fun-ASR
    - https://github.com/alibaba-damo-academy/FunASR
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any

import numpy as np

from modern_asr.core.base import ASRModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.registry import register_model
from modern_asr.core.types import ASRResult, AudioInput, Segment


def _check_deps() -> None:
    try:
        import torch  # noqa: F401
        import funasr  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Fun-ASR requires 'torch' and 'funasr'. "
            "Install with: uv pip install modern-asr[funasr]"
        ) from exc


def _patch_funasr_nano() -> None:
    """Inject missing modules so FunASRNano can be imported/registered.

    funasr 1.3.1 ships FunASRNano but its model.py uses absolute imports
    'from ctc import CTC' and 'from tools.utils import forced_align' that
    fail because the parent package path is not on sys.modules.
    """
    try:
        from funasr.models.fun_asr_nano import ctc as _ctc_mod
        sys.modules.setdefault("ctc", _ctc_mod)
        from funasr.models.fun_asr_nano import tools as _tools_mod
        sys.modules.setdefault("tools", _tools_mod)
        # Trigger registration in funasr tables
        from funasr.models.fun_asr_nano.model import FunASRNano  # noqa: F401
    except Exception:
        pass


class _FunASRBase(ASRModel):
    """Shared logic for Fun-ASR / Paraformer variants."""

    SUPPORTED_LANGUAGES = {"zh", "en", "auto", "multi"}
    SUPPORTED_MODES = {"transcribe"}
    REQUIREMENTS = ["torch", "torchaudio", "funasr", "modelscope"]

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
        defaults = {
            "funasr-nano": "FunAudioLLM/Fun-ASR-Nano-2512",
            "paraformer-zh": "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1",
            "paraformer-large": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        }
        return defaults.get(self.config.model_id, f"iic/{self.config.model_id}")

    def load(self) -> None:
        _check_deps()
        from funasr import AutoModel

        if self.config.model_id == "funasr-nano":
            _patch_funasr_nano()

        device = self._resolve_device()
        self._model = AutoModel(
            model=self._model_path,
            device=device,
            trust_remote_code=True,
        )
        self._is_loaded = True

    def transcribe(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        self._ensure_loaded()
        language = kwargs.get("language", self.config.language or "auto")

        if self.config.model_id == "funasr-nano":
            return self._transcribe_nano(audio, language)

        waveform = self._to_waveform(audio)
        res = self._model.generate(  # type: ignore[union-attr]
            input=waveform,
            cache={},
            language=language,
            batch_size_s=60,
        )

        if isinstance(res, list) and len(res) > 0:
            item = res[0]
            text = item.get("text", "")
        else:
            text = ""

        return ASRResult(
            text=text,
            segments=[Segment(text=text)],
            language=self.config.language,
            model_id=self.model_id,
            extra={"raw": res},
        )

    def _transcribe_nano(self, audio: AudioInput, language: str) -> ASRResult:
        """Chunk long audio to avoid OOM on consumer GPUs."""
        import librosa
        import soundfile as sf

        # Load and resample to 16kHz
        if audio.is_array():
            arr = audio.data
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)
            sr = audio.sample_rate
            if sr != 16000:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        else:
            arr, _ = librosa.load(str(audio.data), sr=16000)

        chunk_samples = 25 * 16000  # 25-second chunks
        total = len(arr)
        texts: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(0, total, chunk_samples):
                chunk = arr[i : i + chunk_samples]
                wav_path = os.path.join(tmpdir, f"chunk_{i}.wav")
                sf.write(wav_path, chunk, 16000)
                res = self._model.generate(  # type: ignore[union-attr]
                    input=wav_path,
                    batch_size_s=60,
                    language=language,
                )
                if isinstance(res, list) and len(res) > 0:
                    texts.append(res[0].get("text", ""))

        full_text = "".join(texts)
        return ASRResult(
            text=full_text,
            segments=[Segment(text=full_text)],
            language=language,
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

    def _resolve_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@register_model("funasr-nano")
class FunASRNano(_FunASRBase):
    """Fun-ASR-Nano: 0.8B params, trained on tens of millions of hours."""

    MODEL_CARD = "https://huggingface.co/iic/Fun-ASR-Nano"

    @property
    def model_id(self) -> str:
        return "funasr-nano"


@register_model("paraformer-zh")
class ParaformerZH(_FunASRBase):
    """Paraformer-zh: 220M, mature streaming ASR from DAMO Academy."""

    MODEL_CARD = "https://modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1"

    @property
    def model_id(self) -> str:
        return "paraformer-zh"


@register_model("paraformer-large")
class ParaformerLarge(_FunASRBase):
    """Paraformer-Large: large variant with VAD and punctuation."""

    MODEL_CARD = "https://modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

    @property
    def model_id(self) -> str:
        return "paraformer-large"
