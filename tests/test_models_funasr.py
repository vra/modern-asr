"""Mock tests for Fun-ASR model adapters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from modern_asr.core.config import ModelConfig
from modern_asr.core.types import AudioInput
from modern_asr.models.funasr_model import FunASRNano, ParaformerZH


class TestFunASRNano:
    @patch("funasr.AutoModel")
    def test_load(self, mock_automodel):
        mock_model = MagicMock()
        mock_automodel.return_value = mock_model

        cfg = ModelConfig(model_id="funasr-nano")
        m = FunASRNano(cfg)
        m.load()

        assert m.is_loaded
        mock_automodel.assert_called_once()

    @patch("funasr.AutoModel")
    def test_transcribe(self, mock_automodel):
        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "测试结果"}]
        mock_automodel.return_value = mock_model

        cfg = ModelConfig(model_id="funasr-nano")
        m = FunASRNano(cfg)
        m.load()

        arr = np.ones(16000 * 2, dtype=np.float32)
        audio = AudioInput(data=arr, sample_rate=16000)
        result = m.transcribe(audio, language="zh")

        assert result.text == "测试结果"
        assert result.model_id == "funasr-nano"


class TestParaformerZH:
    @patch("funasr.AutoModel")
    def test_load(self, mock_automodel):
        mock_model = MagicMock()
        mock_automodel.return_value = mock_model

        cfg = ModelConfig(model_id="paraformer-zh")
        m = ParaformerZH(cfg)
        m.load()

        assert m.is_loaded
        assert m.model_id == "paraformer-zh"
