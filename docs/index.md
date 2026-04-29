# Modern ASR

A **unified, extensible, and future-proof** Python toolkit for locally running state-of-the-art LLM-based Automatic Speech Recognition (ASR) models.

---

## What is Modern ASR?

Modern ASR integrates cutting-edge open-source speech recognition models from leading research labs — **Alibaba**, **Xiaomi**, **Xiaohongshu**, **NVIDIA**, **Zhipu AI**, **OpenAI**, **IBM**, and more — behind a single, clean API.

Every model runs **locally on your hardware**. No cloud APIs, no data leaves your machine.

<div class="grid cards" markdown>

- :material-lightning-bolt: **23 Models** — From 39M to 8B parameters, covering Whisper, SenseVoice, Qwen, MiMo, and more
- :material-swap-horizontal: **Hot-swap** — Switch models at runtime without restarting your application
- :material-earth: **Multi-Language** — Mandarin, Cantonese, 22 Chinese dialects, English, Japanese, Korean, 50+ languages
- :material-server: **Local-First** — All inference happens on-device. No network required after download
- :material-puzzle: **Plugin Architecture** — Add new models with a single `@register_model` decorator
- :material-code-braces: **Multiple Backends** — Transformers, vLLM, ONNX Runtime

</div>

---

## Quick Start

```bash
# Create a Python 3.12 environment (recommended for latest models)
uv venv --python python3.12 .venv
source .venv/bin/activate

# Install with the models you need
uv pip install "modern-asr[sensevoice,whisper,qwen-asr]"
```

```python
from modern_asr import ASRPipeline

# Load any model by ID
pipe = ASRPipeline("sensevoice-small")
result = pipe("audio.wav", language="zh")
print(result.text)

# Switch to another model instantly
pipe.switch_model("qwen3-asr-0.6b")
result = pipe("audio.wav", language="zh")
```

---

## Supported Models

| Family | Models | Params | Best For |
|--------|--------|--------|----------|
| **Whisper** | tiny, base, small, medium, large-v3, large-v3-turbo | 39M–1.5B | General-purpose, multilingual |
| **SenseVoice** | small, large | — | Chinese-centric, emotion/event detection |
| **FireRedASR** | AED, LLM | — | High-accuracy Chinese named entities |
| **Qwen3-ASR** | 0.6B, 1.7B | 0.6B–1.7B | 52 languages, 22 Chinese dialects |
| **Qwen2.5-Omni** | 7B | 7B | Multimodal (audio + text + image) |
| **Fun-ASR-Nano** | nano | 0.8B | Speed, tens of millions of hours training |
| **Paraformer** | zh, large | 220M | Streaming, mature Chinese ASR |
| **MiMo-V2.5-ASR** | v2.5 | 8B | Dialects, code-switching, noise, songs |
| **GLM-ASR** | nano-2512 | 1.5B | Chinese, English |
| **Canary-Qwen** | 2.5B | 2.5B | NVIDIA-optimized multilingual |
| **Granite-Speech** | 3.3-8B | 8B | IBM enterprise multilingual |
| **LLaMA-Omni** | 8B | 8B | Speech-to-text conversation |
| **MiDashengLM** | 7B | 7B | Xiaomi general audio understanding |
| **Moonshine** | tiny | — | Edge deployment, ONNX |

---

## Design Philosophy

### Universal → No Model Left Behind

Modern ASR is built on two core abstractions:

1. **`ASRModel`** — The base class every adapter implements. It handles audio I/O, device placement, chunked inference, and result formatting.
2. **`AudioLLMModel`** — A reusable base for the increasingly common "audio encoder + LLM decoder" architecture. New Audio-LLM models often need only **5 lines of configuration**.

```python
@register_model("my-audio-llm")
class MyAudioLLM(AudioLLMModel):
    HF_PATH = "org/MyAudioLLM-1B"
    MODEL_CLS = "transformers.AutoModelForSpeechSeq2Seq"
    PROCESSOR_CLS = "transformers.AutoProcessor"
    CHUNK_DURATION = 30.0
```

### Local-First → Privacy & Control

All models download weights to your local disk and run entirely on your CPU/GPU. No API keys, no rate limits, no data leakage.

### Future-Proof → Pluggable by Design

New architectures appear every month. Modern ASR handles them through:

- **Dynamic class imports** — `MODEL_CLS = "transformers.SomeNewModel"`
- **`trust_remote_code`** — Built-in support for custom modeling files on HuggingFace
- **Custom repo wrappers** — MiMo, FireRedASR, and others load from cloned GitHub repos
- **Backend abstraction** — Swap between Transformers, vLLM, and ONNX without changing model code
