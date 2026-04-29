"""Mock tests for Qwen3-ASR model adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from modern_asr.core.config import ModelConfig
from modern_asr.core.types import AudioInput
from modern_asr.models.qwen_asr import Qwen3ASR06B


class TestQwen3ASR:
    @patch("qwen_asr.Qwen3ASRModel")
    def test_load(self, mock_cls):
        mock_instance = MagicMock()
        mock_cls.from_pretrained.return_value = mock_instance

        cfg = ModelConfig(model_id="qwen3-asr-0.6b")
        m = Qwen3ASR06B(cfg)
        m.load()

        assert m.is_loaded
        mock_cls.from_pretrained.assert_called_once()

    @patch("qwen_asr.Qwen3ASRModel")
    def test_transcribe(self, mock_cls):
        mock_result = MagicMock()
        mock_result.text = " hello world "
        mock_result.language = "English"

        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = [mock_result]
        mock_cls.from_pretrained.return_value = mock_instance

        cfg = ModelConfig(model_id="qwen3-asr-0.6b")
        m = Qwen3ASR06B(cfg)
        m.load()

        arr = np.ones(16000 * 2, dtype=np.float32)
        audio = AudioInput(data=arr, sample_rate=16000)
        result = m.transcribe(audio, language="en")

        assert result.text == "hello world"
        assert result.model_id == "qwen3-asr-0.6b"

    def test_language_map(self):
        cfg = ModelConfig(model_id="qwen3-asr-0.6b")
        m = Qwen3ASR06B(cfg)
        assert m._LANGUAGE_MAP["zh"] == "Chinese"
        assert m._LANGUAGE_MAP["en"] == "English"
