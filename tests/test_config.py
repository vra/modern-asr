"""Tests for configuration classes."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from modern_asr.core.config import BackendConfig, ModelConfig, PipelineConfig


# ------------------------------------------------------------------ #
# BackendConfig
# ------------------------------------------------------------------ #


class TestBackendConfig:
    def test_defaults(self):
        b = BackendConfig()
        assert b.name == "auto"
        assert b.device == "auto"
        assert b.dtype == "auto"

    def test_custom_values(self):
        b = BackendConfig(device="cuda:0", dtype="float16", batch_size=8)
        assert b.device == "cuda:0"
        assert b.dtype == "float16"
        assert b.batch_size == 8


# ------------------------------------------------------------------ #
# ModelConfig
# ------------------------------------------------------------------ #


class TestModelConfig:
    def test_defaults(self):
        m = ModelConfig(model_id="test-model")
        assert m.model_id == "test-model"
        assert m.language == "auto"

    def test_model_path_resolution(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            m = ModelConfig(model_id="test", model_path=f.name)
            assert isinstance(m.model_path, Path)
            assert m.model_path == Path(f.name).resolve()


# ------------------------------------------------------------------ #
# PipelineConfig
# ------------------------------------------------------------------ #


class TestPipelineConfig:
    def test_from_yaml(self):
        data = {
            "default_backend": {"name": "transformers", "device": "cuda"},
            "models": {
                "whisper-small": {"model_id": "whisper-small", "language": "en"},
                "sensevoice-small": {"model_id": "sensevoice-small", "language": "zh"},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            config = PipelineConfig.from_yaml(f.name)
            assert config.default_backend.device == "cuda"
            assert "whisper-small" in config.models
            assert config.models["whisper-small"].language == "en"

    def test_empty_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            config = PipelineConfig.from_yaml(f.name)
            assert config.default_backend.name == "auto"
