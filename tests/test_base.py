"""Tests for ASRModel base class utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from modern_asr.core.base import ASRModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.types import AudioInput, ASRResult


class DummyModel(ASRModel):
    """Minimal concrete subclass for testing base utilities."""

    SUPPORTED_LANGUAGES = {"zh", "en"}
    CHUNK_DURATION = 5.0

    def __init__(
        self,
        config: ModelConfig | None = None,
        backend: BackendConfig | None = None,
    ) -> None:
        cfg = config or ModelConfig(model_id="dummy")
        super().__init__(cfg, backend)

    @property
    def model_id(self) -> str:
        return "dummy"

    def load(self) -> None:
        self._is_loaded = True

    def transcribe(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        return ASRResult(text="dummy", segments=[], model_id=self.model_id)


# ------------------------------------------------------------------ #
# Device / dtype resolution
# ------------------------------------------------------------------ #


class TestResolveDevice:
    def test_auto_with_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            m = DummyModel()
            assert m._resolve_device("auto") == "cuda"

    def test_auto_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            m = DummyModel()
            assert m._resolve_device("auto") == "cpu"

    def test_explicit_device(self):
        m = DummyModel()
        assert m._resolve_device("cpu") == "cpu"
        assert m._resolve_device("cuda:1") == "cuda:1"

    def test_backend_fallback(self):
        backend = BackendConfig(device="cuda")
        m = DummyModel(backend=backend)
        with patch("torch.cuda.is_available", return_value=True):
            assert m._resolve_device() == "cuda"


class TestResolveDtype:
    def test_float16(self):
        m = DummyModel()
        assert m._resolve_dtype("float16") == torch.float16

    def test_bfloat16_with_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            m = DummyModel()
            assert m._resolve_dtype("bfloat16") == torch.bfloat16

    def test_bfloat16_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            m = DummyModel()
            assert m._resolve_dtype("bfloat16") == torch.float32

    def test_float32(self):
        m = DummyModel()
        assert m._resolve_dtype("float32") == torch.float32

    def test_auto(self):
        m = DummyModel()
        assert m._resolve_dtype("auto") == torch.float16


# ------------------------------------------------------------------ #
# Audio I/O
# ------------------------------------------------------------------ #


class TestToWaveform:
    def test_from_array(self):
        arr = np.ones(16000, dtype=np.float32)
        audio = AudioInput(data=arr, sample_rate=16000)
        m = DummyModel()
        out = m._to_waveform(audio)
        np.testing.assert_array_equal(out, arr)

    def test_from_file(self):
        arr = np.ones(16000, dtype=np.float32)
        m = DummyModel()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import soundfile as sf

            sf.write(f.name, arr, 16000)
            audio = AudioInput(data=f.name, sample_rate=16000)
            out = m._to_waveform(audio)
            assert len(out) == 16000
            Path(f.name).unlink(missing_ok=True)


class TestSaveTempAudio:
    def test_creates_file(self):
        arr = np.ones(16000, dtype=np.float32)
        audio = AudioInput(data=arr, sample_rate=16000)
        m = DummyModel()
        path = m._save_temp_audio(audio)
        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)


class TestAudioToFile:
    def test_file_input_returns_path(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio = AudioInput(data=f.name, sample_rate=16000)
            m = DummyModel()
            assert m._audio_to_file(audio) == f.name
            Path(f.name).unlink(missing_ok=True)

    def test_array_input_creates_temp(self):
        arr = np.ones(16000, dtype=np.float32)
        audio = AudioInput(data=arr, sample_rate=16000)
        m = DummyModel()
        path = m._audio_to_file(audio)
        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)


# ------------------------------------------------------------------ #
# Chunking
# ------------------------------------------------------------------ #


class TestChunkAudio:
    def test_chunking(self):
        arr = np.ones(16000 * 12, dtype=np.float32)  # 12 seconds
        audio = AudioInput(data=arr, sample_rate=16000)
        m = DummyModel()
        chunks = m._chunk_audio(audio, chunk_duration=5.0, overlap=0.0)
        assert len(chunks) == 3  # 0-5, 5-10, 10-12
        assert all(c.sample_rate == 16000 for c in chunks)

    def test_overlap(self):
        arr = np.ones(16000 * 10, dtype=np.float32)
        audio = AudioInput(data=arr, sample_rate=16000)
        m = DummyModel()
        chunks = m._chunk_audio(audio, chunk_duration=5.0, overlap=2.0)
        # step = 3s, so 0-5, 3-8, 6-10 = 3 chunks
        assert len(chunks) == 3

    def test_negative_duration_raises(self):
        arr = np.ones(16000, dtype=np.float32)
        audio = AudioInput(data=arr, sample_rate=16000)
        m = DummyModel()
        with pytest.raises(ValueError):
            m._chunk_audio(audio, chunk_duration=0.0)


class TestChunkedTranscribe:
    def test_concatenates_results(self):
        arr = np.ones(16000 * 12, dtype=np.float32)
        audio = AudioInput(data=arr, sample_rate=16000)
        m = DummyModel()
        result = m._chunked_transcribe(audio, chunk_duration=5.0)
        # DummyModel.transcribe returns "dummy" for each chunk
        assert "dummy" in result.text
        assert result.model_id == "dummy"


# ------------------------------------------------------------------ #
# Lifecycle
# ------------------------------------------------------------------ #


class TestLifecycle:
    def test_context_manager(self):
        m = DummyModel()
        assert not m.is_loaded
        with m as model:
            assert model.is_loaded
        assert not m.is_loaded

    def test_unload(self):
        m = DummyModel()
        m.load()
        assert m.is_loaded
        m.unload()
        assert not m.is_loaded
        assert m._model is None
