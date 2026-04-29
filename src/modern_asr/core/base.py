"""Abstract base class for all ASR models."""

from __future__ import annotations

import gc
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from modern_asr.core.config import BackendConfig, ModelConfig
    from modern_asr.core.types import ASRResult, AudioInput


class ASRModel(ABC):
    """Abstract base class for all ASR model adapters.

    To add a new model, subclass ``ASRModel``, implement the required methods,
    and decorate the class with ``@register_model("model_id")``.

    The base class provides a rich set of shared utilities for device
    resolution, audio preprocessing, chunked inference, and temporary-file
    handling so that subclasses stay focused on model-specific logic.

    Example::

        @register_model("my-asr")
        class MyASRModel(ASRModel):
            SUPPORTED_LANGUAGES = {"zh", "en"}

            def load(self) -> None:
                ...

            def transcribe(self, audio, **kwargs) -> ASRResult:
                ...

            @property
            def model_id(self) -> str:
                return "my-asr"
    """

    # Override in subclass -------------------------------------------------
    MODEL_CARD: str = ""
    SUPPORTED_LANGUAGES: set[str] = set()
    SUPPORTED_MODES: set[str] = {"transcribe"}
    REQUIREMENTS: list[str] = []

    # Chunking behaviour ---------------------------------------------------
    # 0.0  -> disable automatic chunking (subclass handles it or audio is short)
    # >0.0 -> seconds per chunk for _chunked_transcribe()
    CHUNK_DURATION: float = 0.0

    def __init__(
        self,
        config: ModelConfig,
        backend: BackendConfig | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self._model: Any = None
        self._processor: Any = None
        self._is_loaded = False

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the canonical model identifier."""
        ...

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        """Load model weights and processor into memory.

        Implementations should set ``self._is_loaded = True`` on success.
        """
        ...

    @abstractmethod
    def transcribe(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        """Run speech recognition on ``audio``.

        Args:
            audio: Normalized audio input.
            **kwargs: Additional per-call overrides.

        Returns:
            A unified ``ASRResult``.
        """
        ...

    # ------------------------------------------------------------------ #
    # Optional task methods with default implementations
    # ------------------------------------------------------------------ #

    def translate(
        self,
        audio: AudioInput,
        target_language: str = "en",
        **kwargs: Any,
    ) -> "ASRResult":
        """Translate speech to text in target language."""
        return self.transcribe(
            audio, task="translate", target_language=target_language, **kwargs
        )

    def diarize(self, audio: "AudioInput", **kwargs: Any) -> "ASRResult":
        """Transcribe with speaker diarization."""
        raise NotImplementedError(
            f"Model '{self.model_id}' does not support speaker diarization."
        )

    def detect_emotion(self, audio: "AudioInput", **kwargs: Any) -> "ASRResult":
        """Recognize speech with emotion tags."""
        raise NotImplementedError(
            f"Model '{self.model_id}' does not support emotion detection."
        )

    def detect_events(self, audio: "AudioInput", **kwargs: Any) -> "ASRResult":
        """Detect acoustic events in audio."""
        raise NotImplementedError(
            f"Model '{self.model_id}' does not support acoustic event detection."
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def unload(self) -> None:
        """Release model weights from memory."""
        self._model = None
        self._processor = None
        self._is_loaded = False
        gc.collect()

    def _ensure_loaded(self) -> None:
        if not self._is_loaded:
            self.load()

    def __enter__(self) -> ASRModel:
        self._ensure_loaded()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.unload()

    # ------------------------------------------------------------------ #
    # Shared utilities (device / dtype)
    # ------------------------------------------------------------------ #

    def _resolve_device(self, device: str | None = None) -> str:
        """Resolve device string to a concrete device identifier.

        Args:
            device: Override device. If ``None``, uses ``self.backend.device``
                or falls back to ``"cuda"`` if available.
        """
        import torch

        d = device or (self.backend.device if self.backend else "auto")
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    def _resolve_dtype(self, dtype: str | None = None) -> Any:
        """Resolve dtype string to a concrete ``torch.dtype``.

        Args:
            dtype: Override dtype. If ``None``, uses ``self.backend.dtype``
                or falls back to ``torch.float32``.
        """
        import torch

        dt = dtype or (self.backend.dtype if self.backend else "auto")
        if dt in ("auto", "float16"):
            return torch.float16
        if dt == "bfloat16" and torch.cuda.is_available():
            return torch.bfloat16
        if dt == "float32":
            return torch.float32
        return torch.float32

    # ------------------------------------------------------------------ #
    # Shared utilities (audio I/O)
    # ------------------------------------------------------------------ #

    def _to_waveform(self, audio: "AudioInput") -> np.ndarray:
        """Extract a 1-D numpy waveform from ``AudioInput``.

        Handles array inputs, file paths, and bytes transparently.
        """
        if audio.is_array():
            arr = audio.data
            if isinstance(arr, np.ndarray):
                return arr
            return np.array(arr)
        from modern_asr.utils.audio import load_audio

        loaded = load_audio(str(audio.data))
        # loaded.data may be ndarray or other sequence
        if isinstance(loaded.data, np.ndarray):
            return loaded.data
        return np.array(loaded.data)

    def _save_temp_audio(
        self,
        audio: "AudioInput",
        suffix: str = ".wav",
    ) -> str:
        """Write an array-based ``AudioInput`` to a temporary file.

        Returns:
            Absolute path to the temporary file.  The caller is responsible
            for cleanup if desired (the OS will reclaim it eventually).
        """
        import soundfile as sf

        arr = self._to_waveform(audio)
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        sf.write(path, arr, audio.sample_rate)
        return path

    def _audio_to_file(self, audio: "AudioInput") -> str:
        """Return a file-system path for ``audio``.

        If ``audio`` already points to a file, returns its path.
        If ``audio`` is an in-memory array, writes a temporary WAV file
        and returns that path.
        """
        if audio.is_file():
            return str(audio.data)
        return self._save_temp_audio(audio)

    # ------------------------------------------------------------------ #
    # Shared utilities (chunking)
    # ------------------------------------------------------------------ #

    def _chunk_audio(
        self,
        audio: "AudioInput",
        chunk_duration: float | None = None,
        overlap: float = 0.0,
    ) -> list["AudioInput"]:
        """Split ``audio`` into fixed-duration chunks.

        Args:
            chunk_duration: Seconds per chunk. ``None`` → ``self.CHUNK_DURATION``.
            overlap: Overlap between consecutive chunks in seconds.

        Returns:
            List of ``AudioInput`` chunks.
        """
        from modern_asr.utils.audio import chunk_audio

        if not audio.is_array():
            # Load to memory first so we can slice
            arr = self._to_waveform(audio)
            audio = AudioInput(
                data=arr,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                dtype=audio.dtype,
                source=audio.source,
            )
        dur = chunk_duration if chunk_duration is not None else self.CHUNK_DURATION
        if dur <= 0:
            raise ValueError(
                f"chunk_duration must be positive, got {dur}"
            )
        return chunk_audio(audio, chunk_duration=dur, overlap=overlap)

    def _chunked_transcribe(
        self,
        audio: "AudioInput",
        chunk_duration: float | None = None,
        overlap: float = 0.0,
        **kwargs: Any,
    ) -> "ASRResult":
        """Transcribe long audio by splitting into chunks and concatenating.

        This is a generic fallback.  Subclasses with smarter internal
        chunking (e.g. VAD-based) should override ``transcribe`` directly.

        Args:
            audio: Input audio (any source).
            chunk_duration: Seconds per chunk. ``None`` → ``self.CHUNK_DURATION``.
            overlap: Overlap in seconds (useful for context preservation).
            **kwargs: Forwarded to ``self.transcribe()`` for each chunk.

        Returns:
            A single ``ASRResult`` whose ``text`` is the concatenation of
            all chunk texts.
        """
        from modern_asr.core.types import ASRResult, Segment

        chunks = self._chunk_audio(audio, chunk_duration=chunk_duration, overlap=overlap)
        texts: list[str] = []
        for chunk in chunks:
            result = self.transcribe(chunk, **kwargs)
            texts.append(result.text)

        full_text = " ".join(t.strip() for t in texts if t.strip())
        return ASRResult(
            text=full_text,
            segments=[Segment(text=full_text)],
            language=kwargs.get("language", self.config.language),
            model_id=self.model_id,
        )
