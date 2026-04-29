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
uv pip install modern-asr
```

This installs only the core framework. You will need to add extras for actual model inference.

---

## Model-Specific Extras

### Whisper (OpenAI)
```bash
uv pip install "modern-asr[whisper]"
```

### SenseVoice & Paraformer (Alibaba)
```bash
uv pip install "modern-asr[sensevoice]"
```

### FireRedASR (Xiaohongshu)
```bash
uv pip install "modern-asr[fireredasr]"
```

### Qwen3-ASR (Alibaba)
```bash
uv pip install "modern-asr[qwen-asr]"
```

### MiMo-V2.5-ASR (Xiaomi)
```bash
uv pip install "modern-asr[mimo-asr]"
```

### GLM-ASR (Zhipu AI)
```bash
uv pip install "modern-asr[glm-asr]"
```

### Canary-Qwen (NVIDIA)
```bash
uv pip install "modern-asr[canary]"
```

### Moonshine (Useful Sensors)
```bash
uv pip install "modern-asr[moonshine]"
```

---

## Install Everything

```bash
# All models + all inference backends
uv pip install "modern-asr[all]"
```

---

## Manual Dependencies for Special Cases

### Qwen3-ASR (GitHub-only package)

The `qwen-asr` package is not yet on PyPI. Install from GitHub:

```bash
pip install git+https://github.com/QwenLM/Qwen3-ASR.git
```

### MiMo-V2.5-ASR (Clone official repo)

MiMo requires the official repository code for its audio tokenizer:

```bash
git clone https://github.com/XiaomiMiMo/MiMo-V2.5-ASR.git
cd MiMo-V2.5-ASR
pip install -r requirements.txt

# Download weights
huggingface-cli download XiaomiMiMo/MiMo-V2.5-ASR \
    --local-dir models/MiMo-V2.5-ASR
huggingface-cli download XiaomiMiMo/MiMo-Audio-Tokenizer \
    --local-dir models/MiMo-Audio-Tokenizer
```

### FireRedASR (Clone official repo)

FireRedASR also requires its official repository:

```bash
git clone https://github.com/FireRedTeam/FireRedASR.git /tmp/FireRedASR
cd /tmp/FireRedASR && pip install -e .
```

---

## Development Setup

```bash
git clone https://github.com/your-org/modern-asr.git
cd modern-asr
uv venv --python python3.12 .venv
source .venv/bin/activate
uv pip install -e ".[all,dev]"
```

Run tests:
```bash
pytest tests/
```

Build docs:
```bash
mkdocs serve
```
