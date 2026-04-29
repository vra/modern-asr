# Canary-Qwen

Canary-Qwen is NVIDIA's multilingual ASR model, optimized for production deployment with the NeMo toolkit.

---

## Available Models

| Model ID | Parameters | Best For |
|----------|-----------|----------|
| `canary-qwen-2.5b` | 2.5B | NVIDIA-optimized multilingual ASR |

---

## Installation

```bash
uv pip install "modern-asr[canary]"
```

Or manually:
```bash
pip install nemo-toolkit[asr] torch
```

---

## Usage

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("canary-qwen-2.5b")
result = pipe("audio.wav", language="en")
print(result.text)
```

---

## Supported Languages

`en`, `fr`, `de`, `es`, `auto`, `multi`

---

## Supported Modes

- `transcribe` ✅
- `translate` ✅

---

## Performance Notes

- Built on NVIDIA NeMo framework
- Optimized for NVIDIA GPUs with Tensor Cores
- Supports punctuation and capitalization
