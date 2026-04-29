"""Unified data types for Modern ASR."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np


class RecognitionMode(str, Enum):
    """Recognition mode presets."""

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"
    DIARIZE = "diarize"
    EMOTION = "emotion"
    EVENT = "event"
    MULTI_TASK = "multi_task"


class Language(str, Enum):
    """Common language codes."""

    AUTO = "auto"
    ZH = "zh"
    EN = "en"
    YUE = "yue"  # Cantonese
    JA = "ja"
    KO = "ko"
    FR = "fr"
    DE = "de"
    ES = "es"
    RU = "ru"
    AR = "ar"
    HI = "hi"
    PT = "pt"
    IT = "it"
    NL = "nl"
    PL = "pl"
    TR = "tr"
    VI = "vi"
    TH = "th"
    ID = "id"
    MULTI = "multi"


@dataclass
class AudioInput:
    """Normalized audio input wrapper.

    Accepts file paths, raw bytes, or in-memory numpy arrays.
    The pipeline will normalize to the target sample rate and channel layout.
    """

    data: np.ndarray | Path | str | bytes
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"

    # Optional metadata
    duration: float | None = None
    source: str | None = None

    def is_file(self) -> bool:
        return isinstance(self.data, (str, Path))

    def is_bytes(self) -> bool:
        return isinstance(self.data, bytes)

    def is_array(self) -> bool:
        return isinstance(self.data, np.ndarray)


@dataclass
class WordTimestamp:
    """Word-level timestamp."""

    text: str
    start: float  # seconds
    end: float  # seconds
    confidence: float | None = None
    speaker_id: str | None = None


@dataclass
class Segment:
    """A recognized speech segment."""

    text: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None
    language: str | None = None
    words: list[WordTimestamp] = field(default_factory=list)
    speaker_id: str | None = None

    # Extra model-specific outputs (emotion tags, events, etc.)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ASRResult:
    """Unified ASR result container."""

    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    duration: float | None = None
    model_id: str | None = None

    # Model-specific extra outputs
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Alias for `text`."""
        return self.text
