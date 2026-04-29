# Paraformer

Paraformer is Alibaba DAMO Academy's mature streaming ASR model, widely deployed in production Chinese speech recognition systems.

---

## Available Models

| Model ID | Size | Speed | Best For |
|----------|------|-------|----------|
| `paraformer-zh` | 220M | ⚡⚡⚟ | Streaming Chinese ASR |
| `paraformer-large` | Large | ⚡⚡ | VAD + punctuation + large vocab |

---

## Installation

```bash
uv sync --extra funasr
```

Or manually:
```bash
pip install funasr modelscope torch torchaudio
```

---

## Usage

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("paraformer-zh")
result = pipe("audio.wav", language="zh")
print(result.text)
```

---

## Supported Languages

`zh`, `en`, `auto`, `multi`

---

## Supported Modes

- `transcribe` ✅

---

## Performance Notes

- Paraformer is optimized for **streaming** use cases
- The `paraformer-large` variant includes built-in VAD (Voice Activity Detection) and punctuation restoration
- Weights downloaded from ModelScope
- Very low latency suitable for real-time applications
