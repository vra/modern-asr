"""Mock tests for MiMo-V2.5-ASR model adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from modern_asr.core.config import ModelConfig
from modern_asr.core.types import AudioInput
from modern_asr.models.mimo_asr import MiMoASRV25


class TestMiMoASRV25:
    def test_resolve_model_dir_from_config(self):
        cfg = ModelConfig(model_id="mimo-asr-v2.5", model_path="/custom/path")
        m = MiMoASRV25(cfg)
        assert m._model_dir == "/custom/path"

    def test_language_tags(self):
        cfg = ModelConfig(model_id="mimo-asr-v2.5")
        m = MiMoASRV25(cfg)
        assert m._LANGUAGE_TAGS["zh"] == "<chinese>"
        assert m._LANGUAGE_TAGS["en"] == "<english>"

    def test_transcribe_mocked(self):
        """Test transcribe by mocking out load() internals."""
        cfg = ModelConfig(model_id="mimo-asr-v2.5", model_path="/tmp/fake-model")
        m = MiMoASRV25(cfg)

        # Directly set the model without calling load()
        mock_mimo = MagicMock()
        mock_mimo.asr_sft.return_value = "  hello world  "
        m._model = mock_mimo
        m._is_loaded = True

        arr = np.ones(16000 * 2, dtype=np.float32)
        audio = AudioInput(data=arr, sample_rate=16000)
        result = m.transcribe(audio, language="en")

        assert result.text == "hello world"
        assert result.model_id == "mimo-asr-v2.5"
        mock_mimo.asr_sft.assert_called_once()
