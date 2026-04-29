# SenseVoice

SenseVoice is Alibaba's production-grade ASR system with strong Chinese support, emotion recognition, and acoustic event detection. It uses the FunASR framework.

---

## Available Models

| Model ID | Speed | Chinese | English | Special Features |
|----------|-------|---------|---------|-----------------|
| `sensevoice-small` | ⚡⚡⚟ | ★★★ | ★★★ | Emotion, events, fast |
| `sensevoice-large` | ⚡⚟ | ★★★ | ★★★ | Emotion, events, best quality |

---

## Installation

```bash
uv pip install "modern-asr[sensevoice]"
```

Or manually:
```bash
pip install funasr modelscope torch torchaudio
```

---

## Usage

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("sensevoice-small")

# Basic transcription
result = pipe("audio.wav", language="zh")
print(result.text)

# Detect emotion
result = pipe("audio.wav", language="zh", task="emotion")
print(result.text)  # May include emotion tags like [NEUTRAL], [HAPPY]

# Detect acoustic events
result = pipe("audio.wav", language="zh", task="event")
print(result.text)  # May include event tags like [Speech], [Music]
```

---

## Supported Languages

`zh`, `en`, `yue` (Cantonese), `ja`, `ko`, `auto`, `nospeech`

The `nospeech` language code tells the model to output nothing for silent segments.

---

## Supported Modes

- `transcribe` ✅
- `emotion` ✅
- `event` ✅

---

## Known Issues

**funasr 1.3.1 CMVN dimension mismatch**

SenseVoice Small may fail with `RuntimeError: tensor size 567 vs 560` on funasr 1.3.1. Modern ASR automatically applies a monkey-patch to fix this by reshaping the fbank output from 81-dim to 80-dim before CMVN application.

No user action required — the fix is transparent.

---

## Performance Notes

- Weights are downloaded from ModelScope (Chinese mirror of HuggingFace)
- First load triggers a ~300MB–500MB download
- Small model transcribes 7.5 minutes of audio in ~6 seconds on RTX 4070 Ti
- Supports ITN (Inverse Text Normalization) for converting numbers to digits
