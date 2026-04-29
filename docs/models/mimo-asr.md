# MiMo-V2.5-ASR

MiMo-V2.5-ASR is Xiaomi's state-of-the-art open-source ASR, designed for **complex real-world scenarios**: dialects, code-switching, noisy environments, multi-speaker conversations, and even song lyrics.

It uses a unique **two-stage architecture**:
1. **MiMo-Audio-Tokenizer** — encodes audio into discrete tokens
2. **MiMo-V2.5-ASR** (8B causal LM) — decodes tokens into text

---

## Available Models

| Model ID | Parameters | Best For |
|----------|-----------|----------|
| `mimo-asr-v2.5` | 8B | Dialects, noise, songs, meetings |

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.12 |
| CUDA | ≥ 12.0 |
| flash-attn | ≥ 2.7.4 (strongly recommended) |

---

## Installation

### Step 1: Clone the official repository

```bash
git clone https://github.com/XiaomiMiMo/MiMo-V2.5-ASR.git
cd MiMo-V2.5-ASR
pip install -r requirements.txt
```

### Step 2: Download weights

```bash
huggingface-cli download XiaomiMiMo/MiMo-V2.5-ASR \
    --local-dir models/MiMo-V2.5-ASR
huggingface-cli download XiaomiMiMo/MiMo-Audio-Tokenizer \
    --local-dir models/MiMo-Audio-Tokenizer
```

### Step 3: Install Modern ASR

```bash
uv sync --extra mimo-asr
```

---

## Usage

```python
from modern_asr import ASRPipeline

# Auto-discovers the cloned repo if in current directory
pipe = ASRPipeline("mimo-asr-v2.5")
result = pipe("audio.wav", language="zh")
print(result.text)
```

Or specify paths explicitly:

```python
from modern_asr import ASRPipeline
from modern_asr.core.config import ModelConfig

config = ModelConfig(
    model_id="mimo-asr-v2.5",
    model_path="./MiMo-V2.5-ASR/models/MiMo-V2.5-ASR",
)
pipe = ASRPipeline("mimo-asr-v2.5", model_config=config)
```

---

## Supported Languages

`zh`, `en`, `yue` (Cantonese), `wuu` (Wu/Shanghainese), `nan` (Hokkien), `cmn`, `auto`, `multi`

---

## Supported Modes

- `transcribe` ✅
- `diarize` ✅ (multi-speaker transcription)

---

## Capabilities

| Capability | Description |
|------------|-------------|
| **Chinese Dialects** | Wu, Cantonese, Hokkien, Sichuanese, and more |
| **Code-Switching** | Seamless Chinese-English mixing |
| **Song Recognition** | Lyrics transcription with accompaniment |
| **Noise Robustness** | Heavy noise, far-field, reverberation |
| **Multi-Speaker** | Overlapping conversations, meetings |
| **Native Punctuation** | Prosody-based punctuation generation |

---

## Hardware Requirements

MiMo-V2.5-ASR is an 8B parameter model:

| Precision | VRAM | Notes |
|-----------|------|-------|
| bfloat16 | ~16–18GB | Recommended for RTX 4090 / A100 |
| float16 | ~16–18GB | Similar to bfloat16 |
| int8 / int4 | ~8–10GB | Requires quantization support |

For GPUs with 12GB VRAM, consider:
- CPU offloading via `device_map="auto"`
- 4-bit quantization with `bitsandbytes`
