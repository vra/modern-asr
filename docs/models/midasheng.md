# MiDashengLM

MiDashengLM is Xiaomi's general audio understanding model, capable of ASR plus broader audio comprehension tasks.

---

## Available Models

| Model ID | Parameters | Best For |
|----------|-----------|----------|
| `midashenglm-7b` | 7B | General audio understanding |

---

## Installation

```bash
uv sync --extra transformers
```

---

## Usage

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("midashenglm-7b")
result = pipe("audio.wav", language="zh")
print(result.text)
```

---

## Supported Languages

`zh`, `en`, `auto`, `multi`

---

## Supported Modes

- `transcribe` ✅
- `multi_task` ✅

---

## Hardware Requirements

| Precision | VRAM |
|-----------|------|
| float16 | ~14GB |
| bfloat16 | ~14GB |
