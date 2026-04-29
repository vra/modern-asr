# Qwen2.5-Omni

Qwen2.5-Omni is Alibaba's fully multimodal model capable of **audio understanding, speech recognition, text generation, and image reasoning** in a single architecture.

---

## Available Models

| Model ID | Parameters | Best For |
|----------|-----------|----------|
| `qwen2.5-omni-7b` | 7B | Multimodal ASR, audio reasoning |

---

## Installation

```bash
uv sync --extra qwen-asr
```

Requires `transformers >= 4.50` for `Qwen2_5OmniModel` and `Qwen2_5OmniProcessor` support.

---

## Usage

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("qwen2.5-omni-7b")
result = pipe("audio.wav", language="zh")
print(result.text)
```

---

## Supported Languages

`zh`, `en`, `yue`, `ja`, `ko`, `auto`, `multi`

---

## Supported Modes

- `transcribe` ✅
- `translate` ✅
- `multi_task` ✅ (audio understanding + reasoning)

---

## Architecture Notes

Qwen2.5-Omni uses a custom processor that accepts both audio and text prompts:

```python
# Internal implementation detail:
processor(
    audios=[waveform],
    text="Transcribe the speech into text.",
    return_tensors="pt",
)
```

Modern ASR handles this transparently through the `AudioLLMModel._build_inputs()` override.

---

## Hardware Requirements

| Precision | VRAM |
|-----------|------|
| float16 | ~14GB |
| bfloat16 | ~14GB |
| int8 (if supported) | ~8GB |

For GPUs with <14GB VRAM, consider CPU offloading or quantization.
