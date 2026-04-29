"""vLLM backend for accelerated LLM-based ASR inference."""

from __future__ import annotations

from typing import Any

from modern_asr.backends.base import InferenceBackend


class VLLMBackend(InferenceBackend):
    """Backend powered by ``vLLM`` (mainly for text-generation/decoding stages).

    Note:
        vLLM is primarily designed for text LLMs. For encoder-decoder ASR
        models the ``TransformersBackend`` is usually more appropriate.
        This backend is useful when the ASR architecture exposes a pure
        decoder-style generation interface.
    """

    def __init__(self, device: str = "auto", dtype: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, dtype=dtype, **kwargs)
        self._llm: Any = None

    def load(self, model_path: str, **kwargs: Any) -> Any:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "vLLM is required for VLLMBackend. Install with: uv sync --extra vllm"
            ) from exc

        dtype_map = {
            "auto": "auto",
            "float16": "half",
            "bfloat16": "bfloat16",
            "float32": "float",
        }
        dtype = dtype_map.get(self.dtype, "auto")
        gpu_mem = kwargs.pop("gpu_memory_utilization", 0.9)

        self._llm = LLM(
            model=model_path,
            dtype=dtype,
            device="cuda" if self.device == "auto" else self.device,
            gpu_memory_utilization=gpu_mem,
            trust_remote_code=kwargs.pop("trust_remote_code", True),
            **kwargs,
        )
        return self._llm

    def generate(self, inputs: Any, **kwargs: Any) -> Any:
        if self._llm is None:
            raise RuntimeError("LLM not loaded. Call load() first.")
        return self._llm.generate(inputs, **kwargs)

    def unload(self) -> None:
        self._llm = None
        import torch
        torch.cuda.empty_cache()
