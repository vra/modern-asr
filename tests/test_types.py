"""Tests for core types."""

import numpy as np
import pytest

from modern_asr.core.types import ASRResult, AudioInput, Segment


def test_audio_input_from_array():
    arr = np.zeros(16000, dtype=np.float32)
    inp = AudioInput(data=arr, sample_rate=16000)
    assert inp.is_array()
    assert not inp.is_file()


def test_asr_result_segments():
    seg = Segment(text="hello", start=0.0, end=1.0)
    result = ASRResult(text="hello", segments=[seg])
    assert result.full_text == "hello"
    assert len(result.segments) == 1
