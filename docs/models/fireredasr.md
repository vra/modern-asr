# FireRedASR

FireRedASR is Xiaohongshu (小红书)'s open-source ASR system, optimized for high-accuracy Chinese transcription, especially on named entities and domain-specific vocabulary.

Because FireRedASR's HuggingFace configuration is non-standard, Modern ASR loads it through the **official repository** instead of AutoModel.

---

## Available Models

| Model ID | Architecture | Best For |
|----------|-------------|----------|
| `fireredasr-aed` | AED (Attention-Encoder-Decoder) | Accuracy, named entities |
| `fireredasr-llm` | LLM-based | Conversational Chinese |

---

## Installation

### 1. Clone the official repository

```bash
git clone https://github.com/FireRedTeam/FireRedASR.git /tmp/FireRedASR
cd /tmp/FireRedASR && pip install -e .
```

### 2. Install Modern ASR extras

```bash
uv pip install "modern-asr[fireredasr]"
```

---

## Usage

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("fireredasr-aed")
result = pipe("audio.wav", language="zh")
print(result.text)
```

---

## Supported Languages

`zh`, `en`, `yue` (Cantonese), `auto`

---

## Supported Modes

- `transcribe` ✅
- `translate` ✅

---

## Long Audio Handling

FireRedASR does not natively chunk long audio. Modern ASR implements **55-second chunking** to avoid GPU OOM on consumer cards (≤12GB VRAM).

The audio is split into 55-second WAV segments, each transcribed independently, and results are concatenated.

---

## Performance Notes

- First load downloads ~1GB–2GB of weights from HuggingFace
- AED model transcribes 7.5 minutes of audio in ~52 seconds on RTX 4070 Ti
- Excellent accuracy on Chinese named entities (brands, locations, people)
- Use `fireredasr-llm` for more natural, conversational output
