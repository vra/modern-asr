"""IBM Granite Speech adapter.

Models:
    - granite-speech-3.3-8b: Enterprise-grade, Apache-2.0, 5.85% WER

Capabilities:
    - ASR (automatic speech recognition)
    - Speech translation
    - Noise robustness

References:
    - https://huggingface.co/ibm-granite/granite-speech-3.3-8b
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

    ensure_pypi("torch>=2.0")
    ensure_pypi("transformers>=4.40.0")


@register_model("granite-speech-3.3-8b")
class GraniteSpeech33_8B(ASRModel):
    """IBM Granite Speech 3.3 8B: Enterprise open-source ASR, Apache-2.0."""

    MODEL_CARD = "https://huggingface.co/ibm-granite/granite-speech-3.3-8b"
    SUPPORTED_LANGUAGES = {
        "en", "fr", "de", "es", "ja", "zh", "auto", "multi"
    }
    SUPPORTED_MODES = {"transcribe", "translate"}
    REQUIREMENTS = ["torch", "transformers"]

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
        return "ibm-granite/granite-speech-3.3-8b"

    @property
    def model_id(self) -> str:
        return "granite-speech-3.3-8b"

    def load(self) -> None:
        _check_deps()
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch

        backend = self.backend or BackendConfig()
        torch_dtype = self._resolve_torch_dtype(backend.dtype)
        device = self._resolve_device(backend.device)

        self._processor = AutoProcessor.from_pretrained(
            self._hf_path,
            trust_remote_code=True,
        )
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._hf_path,
            torch_dtype=torch_dtype,
            device_map=device if device != "auto" else "auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self._is_loaded = True

    def transcribe(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        self._ensure_loaded()
        import torch

        waveform = self._to_waveform(audio)
        inputs = self._processor(
            waveform,
            sampling_rate=audio.sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}  # type: ignore[union-attr]

        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens or 256),
            "num_beams": kwargs.get("beam_size", self.config.beam_size),
            "do_sample": False,
        }
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **gen_kwargs)  # type: ignore[union-attr]

        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]  # type: ignore[union-attr]
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

    def _resolve_torch_dtype(self, dtype: str) -> Any:
        import torch
        if dtype in ("auto", "float16"):
            return torch.float16
        if dtype == "bfloat16" and (torch.cuda.is_available() or torch.backends.mps.is_available()):
            return torch.bfloat16
        if dtype == "float32":
            return torch.float32
        return torch.float32

    def _resolve_device(self, device: str) -> str:
        import torch
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device
