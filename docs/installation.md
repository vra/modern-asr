# Installation

Modern ASR uses **optional dependency groups** so you only install what you need. The core package itself is lightweight (~50 KB).

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.12 |
| CUDA | 11.8 | 12.1+ |
| PyTorch | 2.0 | 2.3+ |

!!! note "Python 3.9 Users"
    Some newer models (Qwen3-ASR, MiMo-V2.5-ASR) require Python ≥ 3.10 due to upstream dependencies. You can still use Whisper, SenseVoice, and FireRedASR on Python 3.9.

---

## Core Installation

```bash
git clone https://github.com/vra/modern-asr.git
cd modern-asr
uv sync
```

This installs only the core framework. You will need to add extras for actual model inference.

---

## Model-Specific Extras

### Whisper (OpenAI)
```bash
uv sync --extra whisper
```

### SenseVoice & Paraformer (Alibaba)
```bash
uv sync --extra sensevoice
# or Fun-ASR only
uv sync --extra fun-asr
```

### FireRedASR (Xiaohongshu)
```bash
uv sync --extra firered-asr
```

### Qwen3-ASR (Alibaba)
```bash
uv sync --extra qwen-asr
```

### MiMo-V2.5-ASR (Xiaomi)
```bash
uv sync --extra mimo-asr
```

### GLM-ASR (Zhipu AI)
```bash
uv sync --extra glm-asr
```

### Canary-Qwen (NVIDIA)
```bash
uv sync --extra canary-qwen
```

### Moonshine (Useful Sensors)
```bash
uv sync --extra moonshine
```

---

## Install Everything

```bash
# All models + all inference backends
uv sync --all-extras
```

---

## Manual Dependencies for Special Cases

Modern ASR automatically installs missing dependencies, clones git repos, and
downloads HuggingFace weights on first use. You only need manual steps if you
are in an **offline/air-gapped environment**.

### Pre-download everything (offline use)

```bash
# Install all extras upfront
pip install modern-asr[all-models]

# Pre-download model weights (optional, for faster first run)
huggingface-cli download zai-org/GLM-ASR-Nano-2512
huggingface-cli download XiaomiMiMo/MiMo-V2.5-ASR
```

---

## Development Setup

```bash
git clone https://github.com/vra/modern-asr.git
cd modern-asr
uv sync --all-extras
```

Run tests:
```bash
pytest tests/
```

Build docs:
```bash
mkdocs serve
```
