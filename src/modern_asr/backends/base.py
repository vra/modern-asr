"""Abstract inference backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any



from modern_asr.utils.log import get_logger

logger = get_logger(__name__)

class InferenceBackend(ABC):
    """Abstract backend for running model inference.

    Implementations wrap concrete frameworks such as Hugging Face
    ``transformers``, ``vLLM``, or ONNX Runtime.
    """

    def __init__(self, device: str = "auto", dtype: str = "auto", **kwargs: Any) -> None:
        self.device = device
        self.dtype = dtype
        self.extra = kwargs

    @abstractmethod
    def load(self, model_path: str, **kwargs: Any) -> Any:
        """Load a model into the backend and return a handle."""
        ...

    @abstractmethod
    def generate(self, inputs: Any, **kwargs: Any) -> Any:
        """Run generation / inference."""
        ...

    def unload(self) -> None:
        """Release resources."""
        pass
