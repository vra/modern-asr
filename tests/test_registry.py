"""Tests for model registry."""

import pytest

from modern_asr.core.registry import (
    _REGISTRY,
    create_model,
    get_model_class,
    list_models,
    register_model,
)
from modern_asr.core.base import ASRModel
from modern_asr.core.config import ModelConfig


def test_list_models_returns_entries():
    models = list_models()
    assert isinstance(models, list)
    ids = [m["model_id"] for m in models]
    assert "sensevoice-small" in ids
    assert "fireredasr-llm" in ids


def test_get_model_class_existing():
    cls = get_model_class("whisper-large-v3")
    assert issubclass(cls, ASRModel)


def test_get_model_class_missing():
    with pytest.raises(KeyError):
        get_model_class("nonexistent-model")


def test_create_model_without_load():
    cfg = ModelConfig(model_id="whisper-large-v3")
    model = create_model("whisper-large-v3", config=cfg)
    assert isinstance(model, ASRModel)
    assert not model.is_loaded
