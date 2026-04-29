# Moonshine

Moonshine is Useful Sensors' ultra-lightweight ASR model, designed for **edge deployment** with ONNX Runtime.

---

## Available Models

| Model ID | Size | Best For |
|----------|------|----------|
| `moonshine-tiny` | Tiny | Edge devices, ONNX deployment |

---

## Installation

```bash
uv sync --extra moonshine
```

Or manually:
```bash
pip install onnxruntime onnxruntime-gpu
```

---

## Usage

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("moonshine-tiny")
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

## Performance Notes

- Optimized for ONNX Runtime (CPU and GPU)
- Extremely low latency suitable for Raspberry Pi and mobile devices
- English-only
- Falls back to custom ONNX backend if `moonshine` Python package is unavailable
