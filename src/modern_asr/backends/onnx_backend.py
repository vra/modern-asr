"""ONNX Runtime backend for lightweight / edge deployment."""

from __future__ import annotations

from typing import Any

from modern_asr.backends.base import InferenceBackend



from modern_asr.utils.log import get_logger

logger = get_logger(__name__)

class ONNXBackend(InferenceBackend):
    """Backend powered by ONNX Runtime.

    Useful for deploying quantized or exported ASR models on CPU or edge devices.
    """

    def __init__(self, device: str = "auto", dtype: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, dtype=dtype, **kwargs)
        self._session: Any = None
        self._providers: list[str] | None = None

    def load(self, model_path: str, **kwargs: Any) -> Any:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX Runtime is required for ONNXBackend. "
                "Install with: uv sync --extra onnx"
            ) from exc

        providers = kwargs.pop("providers", None)
        if providers is None:
            if self.device.startswith("cuda"):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        self._providers = providers

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        return self._session

    def generate(self, inputs: Any, **kwargs: Any) -> Any:
        if self._session is None:
            raise RuntimeError("ONNX session not loaded. Call load() first.")
        input_names = [inp.name for inp in self._session.get_inputs()]
        if isinstance(inputs, dict):
            feed = {k: v for k, v in inputs.items() if k in input_names}
        else:
            feed = {input_names[0]: inputs}
        return self._session.run(None, feed)

    def unload(self) -> None:
        self._session = None
