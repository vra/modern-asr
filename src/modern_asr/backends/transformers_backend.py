"""Hugging Face Transformers backend."""

from __future__ import annotations

from typing import Any

from modern_asr.backends.base import InferenceBackend


class TransformersBackend(InferenceBackend):
    """Backend powered by Hugging Face ``transformers``."""

    def __init__(self, device: str = "auto", dtype: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, dtype=dtype, **kwargs)
        self._model: Any = None
        self._processor: Any = None
        self._device_final: str = device

    def load(self, model_path: str, **kwargs: Any) -> tuple[Any, Any]:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
        import torch

        torch_dtype = self._resolve_dtype()
        device = self._resolve_device()

        # Try AutoProcessor first, then AutoTokenizer
        try:
            processor = AutoProcessor.from_pretrained(model_path, **kwargs)
        except Exception:
            try:
                processor = AutoTokenizer.from_pretrained(model_path, **kwargs)
            except Exception:
                processor = None

        # Try speech seq2seq, then generic auto model
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device if device != "auto" else "auto",
                low_cpu_mem_usage=True,
                trust_remote_code=kwargs.get("trust_remote_code", True),
            )
        except Exception:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device if device != "auto" else "auto",
                low_cpu_mem_usage=True,
                trust_remote_code=kwargs.get("trust_remote_code", True),
            )

        self._model = model
        self._processor = processor
        self._device_final = device
        return model, processor

    def generate(self, inputs: Any, **kwargs: Any) -> Any:
        import torch
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        with torch.no_grad():
            return self._model.generate(inputs, **kwargs)

    def _resolve_dtype(self) -> Any:
        import torch
        if self.dtype in ("auto", "float16"):
            return torch.float16
        if self.dtype == "bfloat16" and torch.cuda.is_available():
            return torch.bfloat16
        if self.dtype == "float32":
            return torch.float32
        return torch.float32

    def _resolve_device(self) -> str:
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def unload(self) -> None:
        import torch
        self._model = None
        self._processor = None
        torch.cuda.empty_cache()
