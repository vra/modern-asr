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
- 🍎 **Apple Silicon** — MPS (Metal Performance Shaders) support on macOS
- 🐍 **Modern Python** — uv-native packaging, Pydantic configs, rich CLI

---

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install modern-asr
```

**模型依赖和权重会在第一次使用时自动安装** — 你只需要输入模型名字，其余全自动化：

```python
from modern_asr import ASRPipeline

# SenseVoice — 自动安装 funasr、modelscope，自动下载权重
pipe = ASRPipeline("sensevoice-small")

# MiMo-ASR — 自动 clone 官方仓库，自动下载 HF 权重
pipe = ASRPipeline("mimo-asr-v2.5")

# Whisper — 自动安装 openai-whisper，自动下载权重
pipe = ASRPipeline("whisper-small")
```

如果需要预装所有依赖（离线环境）：
```bash
pip install modern-asr[all-models]
```

**Available extras:** `transformers`, `vllm`, `onnx`, `fireredasr`, `sensevoice`, `funasr`, `qwen-asr`, `mimo-asr`, `canary`, `glm-asr`, `whisper`, `moonshine`, `all-models`, `all-backends`, `all`.

### From Source

```bash
# Clone the repository
git clone https://github.com/vra/modern-asr.git
cd modern-asr

# Sync dependencies (recommended)
uv sync --all-extras

# Or install specific extras only
uv sync --extra transformers --extra whisper

# Or just core dependencies
uv sync
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
