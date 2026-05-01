# Modern ASR — Agent Guide

## Project Overview

A unified, extensible Python toolkit for LLM-based Automatic Speech Recognition (ASR).
Designed with a plugin architecture so new models can be added with minimal boilerplate.

## Directory Structure

```
src/modern_asr/
├── __init__.py              # Public API exports, triggers model auto-discovery
├── __main__.py              # python -m modern_asr entrypoint
├── cli.py                   # CLI: modern-asr list / transcribe
├── core/
│   ├── base.py              # ASRModel abstract base class
│   ├── audio_llm.py         # AudioLLMModel reusable base for audio-encoder + LLM-decoder
│   ├── types.py             # AudioInput, ASRResult, Segment, WordTimestamp
│   ├── config.py            # ModelConfig, BackendConfig, PipelineConfig (Pydantic)
│   ├── registry.py          # @register_model decorator, create_model(), list_models()
│   └── pipeline.py          # ASRPipeline high-level API
├── backends/
│   ├── base.py              # InferenceBackend ABC
│   ├── transformers_backend.py
│   ├── vllm_backend.py
│   └── onnx_backend.py
├── models/
│   ├── __init__.py          # auto_discover_models() trigger
│   ├── fireredasr.py        # Xiaohongshu
│   ├── sensevoice.py        # Alibaba
│   ├── qwen_asr.py          # Qwen3-ASR
│   ├── qwen_omni.py         # Qwen2.5-Omni
│   ├── funasr_model.py      # Fun-ASR / Paraformer
│   ├── mimo_asr.py          # Xiaomi
│   ├── midasheng.py         # Xiaomi audio understanding
│   ├── canary_qwen.py       # NVIDIA
│   ├── glm_asr.py           # Zhipu AI
│   ├── granite_speech.py    # IBM
│   ├── whisper_model.py     # OpenAI
│   ├── moonshine.py         # Useful Sensors
│   └── llama_omni.py        # LLaMA-Omni
└── utils/
    ├── audio.py             # load_audio(), chunk_audio()
    ├── auto_install.py      # Auto-install deps, git repos, and HF weights on first use
    └── log.py               # Unified RichHandler logging with ISO timestamps
```

## Adding a New Model

1. Create `src/modern_asr/models/<vendor>_<model>.py`
2. Subclass `ASRModel`
3. Implement `model_id`, `load()`, `transcribe()`
4. Decorate with `@register_model("my-model-id")`
5. Add optional dependency group in `pyproject.toml`

Template:

```python
from modern_asr import ASRModel, register_model, ModelConfig
from modern_asr.core.types import ASRResult, AudioInput

@register_model("my-model")
class MyModel(ASRModel):
    SUPPORTED_LANGUAGES = {"zh", "en"}
    SUPPORTED_MODES = {"transcribe"}

    @property
    def model_id(self) -> str:
        return "my-model"

    def load(self) -> None:
        ...
        self._is_loaded = True

    def transcribe(self, audio: AudioInput, **kwargs) -> ASRResult:
        ...
        return ASRResult(text=..., model_id=self.model_id)
```

## Code Style

- Use `from __future__ import annotations` in every module
- Type hints for all public methods
- Ruff for linting, mypy for type checking
- `rich` for CLI pretty-printing

## Testing

```bash
cd /home/ws/ws/projects/modern-asr
PYTHONPATH=src pytest tests/ -v
```

## Build / Package

```bash
uv sync --all-extras
```
