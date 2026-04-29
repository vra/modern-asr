"""Tests for ASRPipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modern_asr.core.pipeline import ASRPipeline
from modern_asr.core.types import AudioInput


# ------------------------------------------------------------------ #
# Construction
# ------------------------------------------------------------------ #


class TestConstruction:
    def test_init_with_model_id(self):
        pipe = ASRPipeline("whisper-tiny")
        assert pipe._model is not None
        assert pipe._model.model_id == "whisper-tiny"

    def test_init_without_model_id(self):
        pipe = ASRPipeline()
        assert pipe._model is None

    def test_context_manager(self):
        with ASRPipeline("whisper-tiny") as pipe:
            assert pipe._model is not None
            assert pipe._model.is_loaded
        # After exit, model is unloaded
        assert not pipe._model.is_loaded


# ------------------------------------------------------------------ #
# Model switching
# ------------------------------------------------------------------ #


class TestSwitchModel:
    def test_switch_unloads_old(self):
        pipe = ASRPipeline("whisper-tiny")
        old_model = pipe._model
        old_model.load()
        assert old_model.is_loaded

        pipe.switch_model("sensevoice-small")
        assert not old_model.is_loaded
        assert pipe._model.model_id == "sensevoice-small"

    def test_switch_to_same_model_no_op(self):
        pipe = ASRPipeline("whisper-tiny")
        pipe.switch_model("whisper-tiny")
        assert pipe._model.model_id == "whisper-tiny"


# ------------------------------------------------------------------ #
# Input normalization
# ------------------------------------------------------------------ #


class TestInputNormalization:
    def test_string_path(self):
        pipe = ASRPipeline("whisper-tiny")
        with patch.object(pipe._model, "transcribe") as mock_transcribe:
            mock_transcribe.return_value = MagicMock(text="test")
            # Create a real temp file so load_audio succeeds
            arr = np.zeros(16000, dtype=np.float32)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                import soundfile as sf
                sf.write(f.name, arr, 16000)
                pipe(f.name, language="en")
                args = mock_transcribe.call_args
                audio_arg = args[0][0]
                assert isinstance(audio_arg, AudioInput)
                assert audio_arg.is_array()
                Path(f.name).unlink(missing_ok=True)

    def test_numpy_array(self):
        pipe = ASRPipeline("whisper-tiny")
        with patch.object(pipe._model, "transcribe") as mock_transcribe:
            mock_transcribe.return_value = MagicMock(text="test")
            arr = np.zeros(16000, dtype=np.float32)
            pipe(arr, language="en")
            args = mock_transcribe.call_args
            audio_arg = args[0][0]
            assert isinstance(audio_arg, AudioInput)
            assert audio_arg.is_array()
            assert len(audio_arg.data) == 16000

    def test_audio_input_passthrough(self):
        pipe = ASRPipeline("whisper-tiny")
        with patch.object(pipe._model, "transcribe") as mock_transcribe:
            mock_transcribe.return_value = MagicMock(text="test")
            audio = AudioInput(data=np.zeros(16000), sample_rate=16000)
            pipe(audio, language="en")
            args = mock_transcribe.call_args
            assert args[0][0] is audio


# ------------------------------------------------------------------ #
# Task dispatch
# ------------------------------------------------------------------ #


class TestTaskDispatch:
    def test_transcribe_task(self):
        pipe = ASRPipeline("whisper-tiny")
        with patch.object(pipe._model, "transcribe") as mock:
            mock.return_value = MagicMock(text="hello")
            arr = np.zeros(16000, dtype=np.float32)
            result = pipe(arr, task="transcribe", language="en")
            mock.assert_called_once()
            assert result.text == "hello"

    def test_translate_task(self):
        pipe = ASRPipeline("whisper-tiny")
        with patch.object(pipe._model, "translate") as mock:
            mock.return_value = MagicMock(text="bonjour")
            arr = np.zeros(16000, dtype=np.float32)
            result = pipe(arr, task="translate", language="fr")
            mock.assert_called_once()

    def test_diarize_task(self):
        pipe = ASRPipeline("whisper-tiny")
        with patch.object(pipe._model, "diarize") as mock:
            mock.return_value = MagicMock(text="speaker1: hello")
            arr = np.zeros(16000, dtype=np.float32)
            result = pipe(arr, task="diarize")
            mock.assert_called_once()

    def test_unknown_task_falls_back_to_transcribe(self):
        pipe = ASRPipeline("whisper-tiny")
        with patch.object(pipe._model, "transcribe") as mock:
            mock.return_value = MagicMock(text="hello")
            arr = np.zeros(16000, dtype=np.float32)
            result = pipe(arr, task="unknown_task")
            mock.assert_called_once()
            call_kwargs = mock.call_args[1]
            assert call_kwargs.get("task") == "unknown_task"


# ------------------------------------------------------------------ #
# Unload
# ------------------------------------------------------------------ #


class TestUnload:
    def test_unload_releases_model(self):
        pipe = ASRPipeline("whisper-tiny")
        pipe._model.load()
        assert pipe._model.is_loaded
        pipe.unload()
        # Model object still exists but is unloaded
        assert not pipe._model.is_loaded
