# Modern ASR

<p align="center">
  <strong>A unified, extensible, future-proof toolkit for locally running state-of-the-art LLM-based ASR models.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0"></a>
  <a href="https://pypi.org/project/modern-asr/"><img src="https://img.shields.io/pypi/v/modern-asr" alt="PyPI"></a>
</p>

<p align="center">
  <a href="README_zh.md">简体中文</a> ·
  <a href="#features">Features</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#supported-models">Models</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#architecture">Architecture</a>
</p>

---

## ✨ Features

- **🧩 19 Models** — Whisper, SenseVoice, Qwen, MiMo, FireRedASR, GLM-ASR, and more.
- **🔌 Zero-Config Plugin** — Add new models via `@register_model` decorator.
- **🚀 Runtime Hot-Swap** — Switch models without restarting the process.
- **🌍 Multi-Language** — 52 languages, 22 Chinese dialects.
- **🎯 Multi-Task** — Transcription, translation, diarization, emotion, events.
- **💻 Local-First** — All inference on-device. No API keys. No data leaves your machine.
- **🍎 Apple Silicon** — Native MPS (Metal Performance Shaders) support.
- **📦 Auto-Install** — Dependencies, git repos, and HF weights are installed automatically on first use.
- **🐍 Modern Python** — Pydantic configs, rich CLI, ISO-timestamped logging.

---

## 📦 Installation

```bash
pip install modern-asr
```

Dependencies and model weights are **installed automatically** the first time you use a model — just type its name:

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("sensevoice-small")
pipe = ASRPipeline("mimo-asr-v2.5")
pipe = ASRPipeline("whisper-small")
```

For offline/air-gapped environments, pre-install everything:

```bash
pip install modern-asr[all-models]
```

**Available extras:** `transformers`, `vllm`, `onnx`, `firered-asr`, `sensevoice`, `fun-asr`, `qwen-asr`, `mimo-asr`, `canary-qwen`, `glm-asr`, `whisper`, `moonshine`, `all-models`, `all-backends`, `all`.

**Requirements:** Python ≥ 3.10.

---

## 🧩 Supported Models

| Series | Model ID | Params | Languages | Extra |
|--------|----------|--------|-----------|-------|
| **Whisper** (OpenAI) | `whisper-tiny` | 39M | 99+ | `whisper` |
| | `whisper-base` | 74M | 99+ | `whisper` |
| | `whisper-small` | 244M | 99+ | `whisper` |
| | `whisper-medium` | 769M | 99+ | `whisper` |
| | `whisper-large-v3` | 1.5B | 99+ | `whisper` |
| | `whisper-large-v3-turbo` | 809M | 99+ | `whisper` |
| **SenseVoice** (Alibaba) | `sensevoice-small` | 234M | zh/en/ja/ko/yue | `sensevoice` |
| | `sensevoice-large` | — | 50+ | `sensevoice` |
| **Qwen3-ASR** (Alibaba) | `qwen3-asr-0.6b` | 0.6B | 22 dialects | `qwen-asr` |
| | `qwen3-asr-1.7b` | 1.7B | 22 dialects | `qwen-asr` |
| **Qwen2.5-Omni** (Alibaba) | `qwen2.5-omni-7b` | 7B | zh/en | `qwen-asr` |
| **FunASR / Paraformer** (Alibaba) | `funasr-nano` | 0.8B | zh/en | `fun-asr` |
| | `paraformer-zh` | 0.2B | zh | `fun-asr` |
| | `paraformer-large` | 0.7B | zh | `fun-asr` |
| **FireRedASR** (Xiaohongshu) | `fireredasr-aed` | 1.1B | zh | `firered-asr` |
| | `fireredasr-llm` | 8.3B | zh | `firered-asr` |
| **MiMo-ASR** (Xiaomi) | `mimo-asr-v2.5` | 8B | zh/dialects | `mimo-asr` |
| **MiDasheng** (Xiaomi) | `midashenglm-7b` | 7B | audio understanding | `mimo-asr` |
| **Canary-Qwen** (NVIDIA) | `canary-qwen-2.5b` | 2.5B | en/de/fr/es | `canary-qwen` |
| **GLM-ASR** (Zhipu AI) | `glm-asr-nano-2512` | 1.5B | zh/en/yue | `glm-asr` |
| **Granite Speech** (IBM) | `granite-speech-3.3-8b` | 8B | en | `transformers` |
| **Moonshine** (Useful Sensors) | `moonshine-tiny` | 27M | en | `moonshine` |
| **LLaMA-Omni** | `llama-omni-8b` | 8B | zh/en | `transformers` |

```bash
# List all available models
python -m modern_asr list
```

---

## 🚀 Quick Start

```python
from modern_asr import ASRPipeline

# Chinese with SenseVoice
pipe = ASRPipeline("sensevoice-small")
result = pipe("audio.wav", language="zh")
print(result.text)

# Switch to Qwen3-ASR for dialects
pipe.switch_model("qwen3-asr-0.6b")
result = pipe("audio.wav", language="zh")
print(result.text)

# English with Whisper
pipe.switch_model("whisper-small")
result = pipe("audio.wav", language="en")
print(result.text)
```

---

## 🏗️ Architecture

Modern ASR is built on three layers:

1. **ASRPipeline** — Unified user API. Input normalization, task dispatch, model lifecycle.
2. **ASRModel / AudioLLMModel** — Adapter layer. New models often need only **8 lines of config**.
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

The registry auto-discovers it at runtime. That's it.

---

## 📚 Documentation

Full documentation with Material for MkDocs:

```bash
mkdocs serve
```

---

## 🤝 Contributing

See [Contributing Guide](docs/contributing.md) for development setup, code style, and PR checklist.

---

## 📄 License

Apache-2.0

---
