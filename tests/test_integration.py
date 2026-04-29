"""Lightweight integration tests using real model loading (minimal models only).

These tests download and load actual model weights. They are marked with
``@pytest.mark.integration`` so they can be skipped in fast CI runs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from modern_asr import ASRPipeline, list_models
from modern_asr.core.config import BackendConfig
from modern_asr.core.types import AudioInput

pytestmark = pytest.mark.integration


# ------------------------------------------------------------------ #
# Registry / discovery
# ------------------------------------------------------------------ #


class TestRegistryIntegration:
    def test_all_models_registered(self):
        models = list_models()
        ids = {m["model_id"] for m in models}
        expected = {
            "whisper-tiny",
            "sensevoice-small",
            "qwen3-asr-0.6b",
            "funasr-nano",
            "mimo-asr-v2.5",
        }
        assert expected.issubset(ids)


# ------------------------------------------------------------------ #
# Audio I/O round-trip
# ------------------------------------------------------------------ #


class TestAudioRoundTrip:
    def test_load_and_chunk(self):
        from modern_asr.utils.audio import load_audio, chunk_audio

        arr = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 16000 * 3)).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import soundfile as sf

            sf.write(f.name, arr, 16000)
            audio = load_audio(f.name)
            assert audio.sample_rate == 16000
            chunks = chunk_audio(audio, chunk_duration=1.0, overlap=0.0)
            assert len(chunks) == 3
            Path(f.name).unlink(missing_ok=True)


# ------------------------------------------------------------------ #
# Whisper tiny (lightest real model)
# ------------------------------------------------------------------ #


@pytest.mark.slow
class TestWhisperTinyReal:
    def test_load_and_transcribe_silence(self):
        """Load whisper-tiny and transcribe silence."""
        pipe = ASRPipeline("whisper-tiny")
        arr = np.zeros(16000 * 2, dtype=np.float32)  # 2s silence
        audio = AudioInput(data=arr, sample_rate=16000)
        result = pipe(audio, language="en")
        assert isinstance(result.text, str)
        pipe.unload()

    def test_hot_swap(self):
        pipe = ASRPipeline("whisper-tiny")
        pipe.switch_model("whisper-base")
        assert pipe._model.model_id == "whisper-base"
        pipe.unload()
