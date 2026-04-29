# GLM-ASR

GLM-ASR is Zhipu AI's open-source Audio-LLM ASR model, built on the GLM architecture.

---

## Available Models

| Model ID | Parameters | Best For |
|----------|-----------|----------|
| `glm-asr-nano-2512` | 1.5B | Chinese, English, open-source |

---

## Installation

```bash
uv sync --extra glm-asr
```

Or manually:
```bash
pip install transformers sentencepiece torch
```

---

## Usage

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("glm-asr-nano-2512")
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

## Architecture

GLM-ASR uses the standard Audio-LLM pattern and is implemented via `AudioLLMModel`:

```python
HF_PATH = "zai-org/GLM-ASR-Nano-2512"
PROCESSOR_CLS = "transformers.AutoTokenizer"
MODEL_CLS = "transformers.AutoModelForSpeechSeq2Seq"
```

This makes it an excellent reference for how minimal an `AudioLLMModel` subclass can be.

---

## Hardware Requirements

| Precision | VRAM |
|-----------|------|
| float16 | ~3–4GB |
| bfloat16 | ~3–4GB |
| float32 | ~6–8GB |
