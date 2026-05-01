"""LLaMA-Omni adapter.

Models:
    - llama-omni-8b: End-to-end speech interaction model, low-latency
    - llama-omni2-8b: improved version

Capabilities:
    - Speech-to-text (ASR)
    - Speech-to-speech response
    - Low-latency streaming

References:
    - https://github.com/ictnlp/LLaMA-Omni
    - https://huggingface.co/ICTNLP/llama-omni-8b
"""

from __future__ import annotations

from typing import Any

import numpy as np

from modern_asr.core.base import ASRModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.registry import register_model
from modern_asr.core.types import ASRResult, AudioInput, Segment


def _check_deps() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LLaMA-Omni requires 'torch' and 'transformers'. "
            "Install with: uv sync --all-extras"
        ) from exc


@register_model("llama-omni-8b")
class LLaMAOmni8B(ASRModel):
    """LLaMA-Omni-8B: End-to-end speech interaction with low latency."""

    MODEL_CARD = "https://huggingface.co/ICTNLP/llama-omni-8b"
    SUPPORTED_LANGUAGES = {"en", "auto"}
    SUPPORTED_MODES = {"transcribe"}
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
        return "ICTNLP/llama-omni-8b"

    @property
    def model_id(self) -> str:
        return "llama-omni-8b"

    def load(self) -> None:
        _check_deps()
        from transformers import AutoModel, AutoProcessor
        import torch

        backend = self.backend or BackendConfig()
        torch_dtype = self._resolve_torch_dtype(backend.dtype)
        device = self._resolve_device(backend.device)

        self._processor = AutoProcessor.from_pretrained(
            self._hf_path,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
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
            audios=[waveform],
            text="Transcribe the speech into text.",
            sampling_rate=audio.sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens or 256),
            "do_sample": False,
        }
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **gen_kwargs)

        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return ASRResult(
            text=text.strip(),
            segments=[Segment(text=text.strip())],
            language="en",
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
