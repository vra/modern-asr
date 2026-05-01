"""Tests for AudioLLMModel generic base class."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from modern_asr.core.audio_llm import AudioLLMModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.registry import register_model
from modern_asr.core.types import AudioInput


@register_model("test-audio-llm")
class DummyAudioLLM(AudioLLMModel):
    HF_PATH = "test-org/TestModel"
    PROCESSOR_CLS = "transformers.AutoProcessor"
    MODEL_CLS = "transformers.AutoModelForSpeechSeq2Seq"
    LANGUAGE_MAP = {"zh": "<|zh|>", "en": "<|en|>"}
    DEFAULT_MAX_NEW_TOKENS = 128
    CHUNK_DURATION = 0.0

    @property
    def model_id(self) -> str:
        return "test-audio-llm"


# ------------------------------------------------------------------ #
# Path resolution
# ------------------------------------------------------------------ #


class TestPathResolution:
    def test_default_hf_path(self):
        cfg = ModelConfig(model_id="test-audio-llm")
        m = DummyAudioLLM(cfg)
        assert m._hf_path == "test-org/TestModel"

    def test_override_from_config(self):
        cfg = ModelConfig(model_id="test-audio-llm", model_path="/local/model")
        m = DummyAudioLLM(cfg)
        assert m._hf_path == "/local/model"


# ------------------------------------------------------------------ #
# Dynamic imports
# ------------------------------------------------------------------ #


class TestImportCls:
    def test_import_torch_module(self):
        cls = AudioLLMModel._import_cls("torch.nn.Linear")
        import torch.nn as nn

        assert cls is nn.Linear

    def test_import_invalid_path(self):
        with pytest.raises(ValueError, match="Invalid dotted class path"):
            AudioLLMModel._import_cls("NoModulePath")

    def test_import_missing_class(self):
        with pytest.raises(ImportError, match="not found in module"):
            AudioLLMModel._import_cls("torch.nn.NonExistentClass123")


# ------------------------------------------------------------------ #
# Language mapping
# ------------------------------------------------------------------ #


class TestLanguageMap:
    def test_known_code(self):
        cfg = ModelConfig(model_id="test-audio-llm")
        m = DummyAudioLLM(cfg)
        assert m._map_language("zh") == "<|zh|>"
        assert m._map_language("en") == "<|en|>"

    def test_unknown_code_passes_through(self):
        cfg = ModelConfig(model_id="test-audio-llm")
        m = DummyAudioLLM(cfg)
        assert m._map_language("ja") == "ja"

    def test_none(self):
        cfg = ModelConfig(model_id="test-audio-llm")
        m = DummyAudioLLM(cfg)
        assert m._map_language(None) is None


# ------------------------------------------------------------------ #
# Load (mocked)
# ------------------------------------------------------------------ #


class TestLoad:
    def test_load_sets_is_loaded(self):
        mock_proc_cls = MagicMock()
        mock_model_cls = MagicMock()
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_proc
        mock_model_cls.from_pretrained.return_value = mock_model

        cfg = ModelConfig(model_id="test-audio-llm")
        m = DummyAudioLLM(cfg)

        with patch.object(m, "_import_cls", side_effect=[mock_proc_cls, mock_model_cls]):
            m.load()

        assert m.is_loaded
        assert m._processor is mock_proc
        assert m._model is mock_model
        mock_proc_cls.from_pretrained.assert_called_once_with(
            "test-org/TestModel",
            trust_remote_code=True,
        )
        mock_model_cls.from_pretrained.assert_called_once()

    def test_load_with_backend(self):
        mock_proc_cls = MagicMock()
        mock_model_cls = MagicMock()
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_proc
        mock_model_cls.from_pretrained.return_value = mock_model

        backend = BackendConfig(device="cpu", dtype="float32")
        cfg = ModelConfig(model_id="test-audio-llm")
        m = DummyAudioLLM(cfg, backend=backend)

        with patch.object(m, "_import_cls", side_effect=[mock_proc_cls, mock_model_cls]):
            m.load()

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == torch.float32


# ------------------------------------------------------------------ #
# Transcribe (mocked)
# ------------------------------------------------------------------ #


class TestTranscribe:
    def test_transcribe_single(self):
        mock_proc = MagicMock()
        mock_proc.return_value = {"input_features": torch.zeros(1, 80, 3000)}
        mock_proc.batch_decode.return_value = ["hello world"]

        mock_model_instance = MagicMock()
        mock_model_instance.device = "cpu"
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3]])

        cfg = ModelConfig(model_id="test-audio-llm")
        m = DummyAudioLLM(cfg)
        m._processor = mock_proc
        m._model = mock_model_instance
        m._is_loaded = True

        arr = np.ones(16000 * 2, dtype=np.float32)
        audio = AudioInput(data=arr, sample_rate=16000)
        result = m.transcribe(audio, language="en")

        assert result.text == "hello world"
        assert result.model_id == "test-audio-llm"
        mock_model_instance.generate.assert_called_once()

    def test_chunked_for_long_audio(self):
        mock_proc = MagicMock()
        mock_proc.return_value = {"input_features": torch.zeros(1, 80, 3000)}
        mock_proc.batch_decode.return_value = ["chunk"]

        mock_model_instance = MagicMock()
        mock_model_instance.device = "cpu"
        mock_model_instance.generate.return_value = torch.tensor([[1]])

        @register_model("test-chunk-model")
        class ChunkModel(AudioLLMModel):
            HF_PATH = "test-org/Chunk"
            CHUNK_DURATION = 2.0

            @property
            def model_id(self) -> str:
                return "test-chunk-model"

        cfg = ModelConfig(model_id="test-chunk-model")
        m = ChunkModel(cfg)
        m._processor = mock_proc
        m._model = mock_model_instance
        m._is_loaded = True

        arr = np.ones(16000 * 10, dtype=np.float32)  # 10 seconds
        audio = AudioInput(data=arr, sample_rate=16000)
        result = m.transcribe(audio, language="en")

        # Should be chunked into ~5 pieces of 2s each
        assert mock_model_instance.generate.call_count >= 4
        assert "chunk" in result.text


# ------------------------------------------------------------------ #
# Build result
# ------------------------------------------------------------------ #


class TestBuildResult:
    def test_default_result(self):
        cfg = ModelConfig(model_id="test-audio-llm")
        m = DummyAudioLLM(cfg)
        result = m._build_result("  hello  ", language="zh")
        assert result.text == "hello"
        assert result.language == "zh"
        assert result.model_id == "test-audio-llm"
