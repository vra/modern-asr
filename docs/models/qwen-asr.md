# Qwen3-ASR

Qwen3-ASR is Alibaba Qwen Team's latest open-source ASR model, supporting **52 languages** and **22 Chinese dialects** with state-of-the-art accuracy.

Unlike most models in Modern ASR, Qwen3-ASR is distributed via a **dedicated Python package** (`qwen-asr`) rather than standard HuggingFace Auto classes.

---

## Available Models

| Model ID | Parameters | Speed | Best For |
|----------|-----------|-------|----------|
| `qwen3-asr-0.6b` | 0.6B | ⚡⚡⚡ | Speed, real-time |
| `qwen3-asr-1.7b` | 1.7B | ⚡⚡ | Higher accuracy |

---

## Installation

!!! warning "Python ≥ 3.10 Required"
    The `qwen-asr` package requires Python 3.10 or newer due to upstream `accelerate` dependency.

```bash
# Install from GitHub (not yet on PyPI)
pip install git+https://github.com/QwenLM/Qwen3-ASR.git

# Or via Modern ASR extras
uv sync --extra qwen-asr
```

---

## Usage

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("qwen3-asr-0.6b")
result = pipe("audio.wav", language="zh")
print(result.text)

# Switch to larger model
pipe.switch_model("qwen3-asr-1.7b")
result = pipe("audio.wav", language="zh")
```

---

## Supported Languages

52 languages including:

**Chinese**: Mandarin (`zh`), Cantonese (`yue`), Wu/Shanghainese (`wuu`), Hokkien (`nan`), Hakka (`hak`), Gan (`gan`), Xiang (`hsn`), Jin (`cjy`), Central Plains (`czh`), Mandarin (`cmn`)

**Other**: English, Japanese, Korean, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Russian, Thai, Vietnamese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian

!!! note "Language Code Mapping"
    Qwen3-ASR uses full language names internally. Modern ASR maps short codes automatically:
    `zh` → `Chinese`, `en` → `English`, `yue` → `Cantonese`, etc.

---

## Supported Modes

- `transcribe` ✅
- `translate` ✅

---

## Performance Notes

| Model | RTX 4070 Ti (7.5min audio) | VRAM |
|-------|---------------------------|------|
| 0.6B | ~14s | ~3GB |
| 1.7B | ~41s | ~6GB |

- Automatically uses GPU when available
- Supports `float16` and `bfloat16` quantization
- The 0.6B model achieves RTF (Real-Time Factor) of ~0.009 on A100
