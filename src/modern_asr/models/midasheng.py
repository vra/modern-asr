"""MiDashengLM adapter (Xiaomi / 小米).

Models:
    - midashenglm-7b: Audio understanding large model, unified speech/environment sound/music

Capabilities:
    - Speech recognition
    - Environmental sound understanding
    - Music understanding
    - Audio Q&A
    - Low latency, high throughput

References:
    - https://huggingface.co/Xiaomi/MiDashengLM-7B
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

    ensure_pypi("torch>=2.0")
    ensure_pypi("transformers>=4.40.0")


@register_model("midashenglm-7b")
class MiDashengLM7B(ASRModel):
    """MiDashengLM-7B: Xiaomi's unified audio understanding model."""

    MODEL_CARD = "https://huggingface.co/mispeech/midashenglm-7b-1021-fp32"
    SUPPORTED_LANGUAGES = {"zh", "en", "auto", "multi"}
    SUPPORTED_MODES = {"transcribe", "multi_task"}
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
        # Upstream model is hosted under the mispeech org rather than Xiaomi
        return "mispeech/midashenglm-7b-1021-fp32"

    @property
    def model_id(self) -> str:
        return "midashenglm-7b"

    def load(self) -> None:
        logger.info("Loading %s", self.model_id)

        _check_deps()
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch

        backend = self.backend or BackendConfig()
        torch_dtype = self._resolve_torch_dtype(backend.dtype)
        device = self._resolve_device(backend.device)

        self._processor = AutoProcessor.from_pretrained(
            self._hf_path,
            trust_remote_code=True,
        )
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._hf_path,
                torch_dtype=torch_dtype,
                device_map=device if device != "auto" else "auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except RuntimeError as exc:
            if "meta" in str(exc):
                raise RuntimeError(
                    f"{self.model_id} failed to load due to a device compatibility issue "
                    "between the model's custom code and the installed PyTorch/transformers "
                    "version. This typically occurs when custom model initialization creates "
                    "tensors on CPU while transformers expects the meta device. "
                    f"Original error: {exc}"
                ) from exc
            raise
        self._is_loaded = True

    def transcribe(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        logger.info("Transcribing with %s", self.model_id)

        self._ensure_loaded()
        import torch

        waveform = self._to_waveform(audio)
        # MiDashengLM typically uses a chat template with audio
        prompt = kwargs.get("prompt", "Please transcribe the audio content.")
        inputs = self._processor(
            audios=[waveform],
            text=prompt,
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
