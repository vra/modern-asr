# Modern ASR

A **unified, extensible, and future-proof** Python toolkit for locally running state-of-the-art LLM-based Automatic Speech Recognition (ASR) models.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

---

## ✨ Features

- 🧩 **23 Models** — Whisper, SenseVoice, Qwen, MiMo, FireRedASR, GLM-ASR, and more
- 🔌 **Plugin Architecture** — Add new models with `@register_model` decorator
- 🚀 **Hot-Swap** — Switch models at runtime without restarting
- 🌍 **Multi-Language** — 52 languages, 22 Chinese dialects
- 🎯 **Multi-Task** — Transcription, translation, diarization, emotion, events
- 💻 **Local-First** — All inference on-device. No APIs. No data leaves your machine.
- 🐍 **Modern Python** — uv-native packaging, Pydantic configs, rich CLI

---

## 📦 Installation

```bash
# Using uv (recommended)
uv pip install modern-asr

# With specific model support
uv pip install "modern-asr[sensevoice,fireredasr,whisper]"

# All models + all backends
uv pip install "modern-asr[all]"
```

**Python 3.10+ recommended.** Some models (Qwen3-ASR, MiMo) require Python ≥ 3.10.

---

## 🚀 Quick Start

```python
from modern_asr import ASRPipeline

# Transcribe with SenseVoice (Alibaba)
pipe = ASRPipeline("sensevoice-small")
result = pipe("audio.wav", language="zh")
print(result.text)

# Switch to Qwen3-ASR for dialect support
pipe.switch_model("qwen3-asr-0.6b")
result = pipe("audio.wav", language="zh")
print(result.text)

# English with Whisper
pipe.switch_model("whisper-small")
result = pipe("audio.wav", language="en")
```

---

## 📚 Documentation

Full documentation with Material for MkDocs:

```bash
mkdocs serve
```

Or visit: [https://modern-asr.readthedocs.io](https://modern-asr.readthedocs.io)

---

## 🏗️ Architecture

Modern ASR is built on three layers:

1. **ASRPipeline** — Unified user API. Handles input normalization, task dispatch, model lifecycle.
2. **ASRModel / AudioLLMModel** — Adapter layer. New models often need only **8 lines of config** via `AudioLLMModel`.
3. **Backends** — Transformers, vLLM, ONNX Runtime.

### Adding a New Model

```python
from modern_asr.core.audio_llm import AudioLLMModel
from modern_asr.core.registry import register_model

@register_model("my-model-1b")
class MyModel1B(AudioLLMModel):
    HF_PATH = "org/MyModel-1B"
    SUPPORTED_LANGUAGES = {"zh", "en"}
    CHUNK_DURATION = 30.0

    @property
    def model_id(self) -> str:
        return "my-model-1b"
```

That's it. The registry auto-discovers it at runtime.

---

## 🤝 Contributing

See [Contributing Guide](docs/contributing.md) for development setup, code style, and PR checklist.

---

## 📄 License

Apache-2.0
