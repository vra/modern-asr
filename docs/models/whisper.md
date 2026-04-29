# Whisper

OpenAI's Whisper is the foundational open-source ASR model. Modern ASR wraps all official Whisper sizes behind the unified pipeline.

---

## Available Models

| Model ID | Parameters | Speed | Best For |
|----------|-----------|-------|----------|
| `whisper-tiny` | 39M | ⚡⚡⚡ | Prototyping, edge devices |
| `whisper-base` | 74M | ⚡⚡⚡ | Balanced speed/quality |
| `whisper-small` | 244M | ⚡⚡ | Good quality general ASR |
| `whisper-medium` | 769M | ⚡⚡ | High-quality general ASR |
| `whisper-large-v3` | 1.5B | ⚡ | Best Whisper quality |
| `whisper-large-v3-turbo` | 809M | ⚡⚡ | Large quality, faster |

---

## Installation

```bash
uv sync --extra whisper
```

Or manually:
```bash
pip install openai-whisper torch
```

---

## Usage

```python
from modern_asr import ASRPipeline

# English
pipe = ASRPipeline("whisper-small")
result = pipe("audio.wav", language="en")
print(result.text)

# Chinese
pipe.switch_model("whisper-large-v3")
result = pipe("audio.wav", language="zh")
print(result.text)

# Auto-detect language
result = pipe("audio.wav", language="auto")
```

---

## Supported Languages

Whisper supports 99 languages including:

`zh`, `en`, `de`, `es`, `fr`, `ja`, `ko`, `ru`, `yue`, and 90+ more.

Full list: [OpenAI Whisper](https://github.com/openai/whisper#available-models-and-languages)

---

## Supported Modes

- `transcribe` ✅
- `translate` ✅ (to English)

---

## Performance Notes

- Whisper runs entirely on the **official OpenAI package** (`openai-whisper`), not HuggingFace transformers
- GPU acceleration is automatic when CUDA is available
- `float16` is recommended for GPUs with ≥8GB VRAM
- Large v3 Turbo is a distilled version of Large v3 — significantly faster with minimal quality loss

---

## Hardware Requirements

| Model | VRAM (float16) | VRAM (float32) |
|-------|---------------|----------------|
| tiny | ~1GB | ~2GB |
| base | ~1GB | ~2GB |
| small | ~2GB | ~4GB |
| medium | ~5GB | ~10GB |
| large-v3 | ~10GB | ~20GB |
| large-v3-turbo | ~5GB | ~10GB |
