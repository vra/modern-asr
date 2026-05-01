"""FireRedASR adapter (Xiaohongshu / 小红书).

Models:
    - fireredasr-llm: Encoder-Adapter-LLM, 8.3B, highest accuracy
    - fireredasr-aed: Attention-based Encoder-Decoder, 1.1B, fast

References:
    - https://github.com/FireRedTeam/FireRedASR
    - Paper: arXiv:2501.14350
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from modern_asr.core.base import ASRModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.registry import register_model
from modern_asr.core.types import ASRResult, AudioInput, Segment


def _check_deps() -> None:
    from modern_asr.utils.auto_install import ensure_pypi

    ensure_pypi("torch>=2.0")
    ensure_pypi("transformers>=4.40.0")
    ensure_pypi("sentencepiece>=0.2.0")
    ensure_pypi(
        "git+https://github.com/FireRedTeam/FireRedASR.git", "fireredasr"
    )


class _FireRedASRBase(ASRModel):
    """Shared logic for FireRedASR variants."""

    SUPPORTED_LANGUAGES = {"zh", "en", "yue", "auto"}
    SUPPORTED_MODES = {"transcribe", "translate"}
    REQUIREMENTS = ["torch", "transformers", "sentencepiece", "fireredasr"]

    def __init__(
        self,
        config: ModelConfig,
        backend: BackendConfig | None = None,
    ) -> None:
        super().__init__(config, backend)
        self._model_dir = self._resolve_model_dir()
        self._asr_type = "aed" if "aed" in self.model_id else "llm"

    def _resolve_model_dir(self) -> str:
        if self.config.model_path:
            return str(self.config.model_path)
        # Default HuggingFace hub paths – FireRedASR stores weights on HF
        defaults = {
            "fireredasr-llm": "fireredteam/FireRedASR-LLM-L",
            "fireredasr-aed": "fireredteam/FireRedASR-AED-L",
        }
        hf_path = defaults.get(self.config.model_id, f"fireredteam/{self.config.model_id}")
        # Download via huggingface_hub
        from huggingface_hub import snapshot_download
        return snapshot_download(hf_path)

    def load(self) -> None:
        _check_deps()
        from fireredasr.models.fireredasr import FireRedAsr

        # Ensure fireredasr is on PYTHONPATH so submodules can be imported
        firered_pkg = sys.modules["fireredasr"].__path__[0]
        if firered_pkg not in sys.path:
            sys.path.insert(0, os.path.dirname(firered_pkg))

        self._model = FireRedAsr.from_pretrained(self._asr_type, self._model_dir)
        self._is_loaded = True

    def transcribe(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        self._ensure_loaded()

        # FireRedASR-AED max input length is ~60s; LLM is ~30s.
        # Chunk long audio and transcribe piece-by-piece.
        max_dur = 25 if self._asr_type == "llm" else 55
        chunks = self._chunk_audio(audio, max_dur)
        texts = []
        for idx, chunk_path in enumerate(chunks):
            decode_cfg = {
                "use_gpu": 1 if self._resolve_device() in ("cuda", "mps") else 0,
                "beam_size": kwargs.get("beam_size", self.config.beam_size or 3),
                "nbest": 1,
                "decode_max_len": 0,
                "softmax_smoothing": 1.25,
                "aed_length_penalty": 0.6,
                "eos_penalty": 1.0,
            }
            if self._asr_type == "llm":
                decode_cfg.update({
                    "decode_min_len": 0,
                    "repetition_penalty": 3.0,
                    "llm_length_penalty": 1.0,
                    "temperature": 1.0,
                })

            uttid = [f"utt{idx}"]
            wav_list = [chunk_path]
            results = self._model.transcribe(uttid, wav_list, decode_cfg)
            chunk_text = results[0].get("text", "") if results else ""
            texts.append(chunk_text)

        full_text = "".join(texts)
        return ASRResult(
            text=full_text.strip(),
            segments=[Segment(text=full_text.strip())],
            language=self.config.language,
            model_id=self.model_id,
        )

    def _chunk_audio(self, audio: AudioInput, max_dur: float) -> list[str]:
        arr = self._to_waveform(audio)
        sr = 16000
        chunk_samples = int(max_dur * sr)
        paths = []
        for start in range(0, len(arr), chunk_samples):
            end = min(start + chunk_samples, len(arr))
            chunk = arr[start:end]
            tmp = tempfile.mktemp(suffix=".wav")
            import soundfile as sf
            sf.write(tmp, chunk, sr, subtype="PCM_16")
            paths.append(tmp)
        return paths

    def _to_wav_path(self, audio: AudioInput) -> str:
        if audio.is_file():
            path = str(audio.data)
            # FireRedASR requires 16kHz 16-bit PCM WAV
            if path.lower().endswith(".wav"):
                return path
        # Convert to temp WAV
        arr = self._to_waveform(audio)
        tmp_path = tempfile.mktemp(suffix=".wav")
        import soundfile as sf
        sf.write(tmp_path, arr, audio.sample_rate, subtype="PCM_16")
        return tmp_path

    def _to_waveform(self, audio: AudioInput) -> np.ndarray:
        if audio.is_array():
            arr = audio.data
            if isinstance(arr, np.ndarray):
                return arr
        from modern_asr.utils.audio import load_audio
        loaded = load_audio(str(audio.data), target_sr=16000, mono=True)
        return loaded.data  # type: ignore[return-value]

    def _resolve_device(self) -> str:
        import torch
        if self.backend and self.backend.device != "auto":
            return self.backend.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


@register_model("fireredasr-llm")
class FireRedASRLLM(_FireRedASRBase):
    """FireRedASR-LLM: Encoder-Adapter-LLM, 8.3B params, SOTA Mandarin CER 3.05%."""

    MODEL_CARD = "https://huggingface.co/fireredteam/FireRedASR-LLM-L"

    @property
    def model_id(self) -> str:
        return "fireredasr-llm"


@register_model("fireredasr-aed")
class FireRedASRAED(_FireRedASRBase):
    """FireRedASR-AED: Attention-based Encoder-Decoder, 1.1B params, balanced speed/accuracy."""

    MODEL_CARD = "https://huggingface.co/fireredteam/FireRedASR-AED-L"

    @property
    def model_id(self) -> str:
        return "fireredasr-aed"
