# Granite-Speech

Granite-Speech is IBM's enterprise-grade multilingual speech model, part of the Granite model family.

---

## Available Models

| Model ID | Parameters | Best For |
|----------|-----------|----------|
| `granite-speech-3.3-8b` | 8B | IBM enterprise multilingual ASR |

---

## Installation

```bash
uv pip install "modern-asr[transformers]"
```

---

## Usage

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("granite-speech-3.3-8b")
result = pipe("audio.wav", language="en")
print(result.text)
```

---

## Supported Languages

`zh`, `en`, `de`, `es`, `fr`, `ja`, `auto`, `multi`

---

## Supported Modes

- `transcribe` ✅
- `translate` ✅

---

## Hardware Requirements

| Precision | VRAM |
|-----------|------|
| float16 | ~16GB |
| bfloat16 | ~16GB |
