# Quick Start

This guide walks you through common tasks with Modern ASR.

---

## Basic Transcription

```python
from modern_asr import ASRPipeline

# Load a model
pipe = ASRPipeline("sensevoice-small")

# Transcribe an audio file
result = pipe("audio.wav", language="zh")
print(result.text)
```

---

## Switching Models

You can hot-swap models without restarting:

```python
pipe = ASRPipeline("sensevoice-small")
print(pipe("audio.wav").text)

# Switch to Qwen3-ASR for better dialect support
pipe.switch_model("qwen3-asr-0.6b")
print(pipe("audio.wav", language="zh").text)

# Switch to Whisper for English
pipe.switch_model("whisper-small")
print(pipe("audio.wav", language="en").text)
```

---

## Language Selection

Different models accept different language codes:

| Model Family | Code Style | Example |
|--------------|-----------|---------|
| Whisper | ISO-639-1 | `zh`, `en`, `ja` |
| SenseVoice | ISO-639-1 + special | `zh`, `en`, `yue`, `ja`, `nospeech` |
| Qwen3-ASR | Full name | `Chinese`, `English`, `Cantonese` |
| FireRedASR | ISO-639-1 | `zh`, `en`, `yue` |
| Fun-ASR-Nano | ISO-639-1 | `zh`, `en` |

Modern ASR handles the mapping automatically where possible:

```python
# Works for all models — the adapter maps "zh" internally
result = pipe("audio.wav", language="zh")
```

---

## Working with In-Memory Audio

```python
import numpy as np
from modern_asr import ASRPipeline
from modern_asr.core.types import AudioInput

# Load your audio into a numpy array somehow
waveform = np.random.randn(16000 * 10).astype(np.float32)  # 10s dummy audio
audio = AudioInput(data=waveform, sample_rate=16000)

pipe = ASRPipeline("whisper-tiny")
result = pipe(audio, language="en")
print(result.text)
```

---

## Long Audio & Chunking

Some models automatically chunk long audio to avoid OOM:

```python
# Fun-ASR-Nano automatically splits into 25-second chunks
pipe = ASRPipeline("funasr-nano")
result = pipe("podcast_30min.mp3", language="zh")
print(result.text)
```

For models without automatic chunking, you can do it manually:

```python
from modern_asr.utils.audio import load_audio, chunk_audio

audio = load_audio("long_interview.wav")
chunks = chunk_audio(audio, chunk_duration=30.0, overlap=2.0)

pipe = ASRPipeline("whisper-small")
texts = [pipe(chunk, language="en").text for chunk in chunks]
full_text = " ".join(texts)
```

---

## GPU & Precision Control

```python
from modern_asr import ASRPipeline
from modern_asr.core.config import BackendConfig

backend = BackendConfig(device="cuda", dtype="float16")
pipe = ASRPipeline("qwen3-asr-1.7b", backend=backend)
result = pipe("audio.wav")
```

| dtype | VRAM | Speed | Quality |
|-------|------|-------|---------|
| `float32` | 2× | Baseline | Best |
| `float16` | 1× | Fast | Excellent |
| `bfloat16` | 1× | Fast | Excellent (Ampere+) |

---

## Context Manager (Auto Cleanup)

```python
from modern_asr import ASRPipeline

with ASRPipeline("whisper-large-v3") as pipe:
    result = pipe("audio.wav")
    print(result.text)
# Model is automatically unloaded here
```

---

## Batch Processing

```python
from modern_asr import ASRPipeline
from pathlib import Path

pipe = ASRPipeline("sensevoice-small")
audio_files = list(Path("./podcasts").glob("*.mp3"))

for path in audio_files:
    result = pipe(str(path), language="zh")
    print(f"{path.name}: {result.text[:100]}...")
```

---

## Next Steps

- [Architecture Overview](architecture.md) — Understand the plugin system
- [Model Documentation](models/index.md) — Detailed guides for every model
- [API Reference](api/pipeline.md) — Complete API documentation
