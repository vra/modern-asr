"""Tests for audio utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from modern_asr.core.types import AudioInput
from modern_asr.utils.audio import chunk_audio, load_audio, _resample_simple


# ------------------------------------------------------------------ #
# load_audio
# ------------------------------------------------------------------ #


class TestLoadAudio:
    def test_load_wav_file(self):
        arr = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import soundfile as sf

            sf.write(f.name, arr, 16000)
            audio = load_audio(f.name)
            assert audio.is_array()
            assert audio.sample_rate == 16000
            assert audio.channels == 1
            Path(f.name).unlink(missing_ok=True)

    def test_resampling(self):
        arr = np.ones(32000, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import soundfile as sf

            sf.write(f.name, arr, 32000)
            audio = load_audio(f.name, target_sr=16000)
            assert audio.sample_rate == 16000
            Path(f.name).unlink(missing_ok=True)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/audio.wav")


# ------------------------------------------------------------------ #
# chunk_audio
# ------------------------------------------------------------------ #


class TestChunkAudio:
    def test_basic_chunking(self):
        arr = np.ones(16000 * 10, dtype=np.float32)
        audio = AudioInput(data=arr, sample_rate=16000)
        chunks = chunk_audio(audio, chunk_duration=3.0, overlap=0.0)
        assert len(chunks) == 4  # 0-3, 3-6, 6-9, 9-10
        assert all(isinstance(c, AudioInput) for c in chunks)

    def test_with_overlap(self):
        arr = np.ones(16000 * 10, dtype=np.float32)
        audio = AudioInput(data=arr, sample_rate=16000)
        chunks = chunk_audio(audio, chunk_duration=5.0, overlap=2.0)
        # step = 3s: 0-5, 3-8, 6-10
        assert len(chunks) == 3

    def test_requires_array(self):
        audio = AudioInput(data="/fake/path.wav", sample_rate=16000)
        with pytest.raises(ValueError, match="requires an array-based AudioInput"):
            chunk_audio(audio, chunk_duration=3.0)

    def test_short_audio_single_chunk(self):
        arr = np.ones(16000 * 2, dtype=np.float32)
        audio = AudioInput(data=arr, sample_rate=16000)
        chunks = chunk_audio(audio, chunk_duration=5.0)
        assert len(chunks) == 1


# ------------------------------------------------------------------ #
# _resample_simple
# ------------------------------------------------------------------ #


class TestResampleSimple:
    def test_upsample(self):
        arr = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        out = _resample_simple(arr, orig_sr=1, target_sr=2)
        assert len(out) == 6

    def test_downsample(self):
        arr = np.arange(10, dtype=np.float32)
        out = _resample_simple(arr, orig_sr=10, target_sr=5)
        assert len(out) == 5

    def test_no_change(self):
        arr = np.ones(10, dtype=np.float32)
        out = _resample_simple(arr, orig_sr=16_000, target_sr=16_000)
        np.testing.assert_array_equal(out, arr)
