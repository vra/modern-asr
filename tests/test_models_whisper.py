"""Mock tests for Whisper model adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modern_asr.core.config import ModelConfig
from modern_asr.core.registry import get_model_class
from modern_asr.core.types import AudioInput


class TestWhisperAdapter:
    @patch("whisper.load_model")
    def test_load_tiny(self, mock_load):
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        cls = get_model_class("whisper-tiny")
        cfg = ModelConfig(model_id="whisper-tiny")
        m = cls(cfg)
        m.load()

        assert m.is_loaded
        mock_load.assert_called_once_with("tiny")

    @patch("whisper.load_model")
    def test_transcribe(self, mock_load):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": " hello world "}
        mock_load.return_value = mock_model

        cls = get_model_class("whisper-tiny")
        cfg = ModelConfig(model_id="whisper-tiny")
        m = cls(cfg)
        m.load()

        arr = np.ones(16000 * 2, dtype=np.float32)
        audio = AudioInput(data=arr, sample_rate=16000)
        result = m.transcribe(audio, language="en")

        assert result.text == "hello world"
        assert result.model_id == "whisper-tiny"
