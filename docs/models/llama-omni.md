# LLaMA-Omni

LLaMA-Omni is a speech-to-text conversation model based on LLaMA, capable of understanding spoken input and generating text responses.

---

## Available Models

| Model ID | Parameters | Best For |
|----------|-----------|----------|
| `llama-omni-8b` | 8B | Speech-to-text chat, conversation |

---

## Installation

```bash
uv sync --extra transformers
```

---

## Usage

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("llama-omni-8b")
result = pipe("audio.wav", language="en")
print(result.text)
```

---

## Supported Languages

`en`, `auto`

---

## Supported Modes

- `transcribe` ✅

---

## Hardware Requirements

| Precision | VRAM |
|-----------|------|
| float16 | ~16GB |
| bfloat16 | ~16GB |
