# Fun-ASR-Nano

Fun-ASR-Nano is Alibaba Tongyi Lab's lightweight Audio-LLM ASR model (0.8B parameters), trained on **tens of millions of hours** of audio data. It delivers excellent accuracy at high speed.

---

## Available Models

| Model ID | Parameters | Speed | Best For |
|----------|-----------|-------|----------|
| `funasr-nano` | 0.8B | ⚡⚡⚡ | Accuracy + speed balance |

---

## Installation

```bash
uv sync --extra fun-asr
```

Or manually:
```bash
pip install funasr modelscope torch torchaudio
```

---

## Usage

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("funasr-nano")
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

## Long Audio Handling

Fun-ASR-Nano's attention mechanism consumes significant VRAM for long sequences. Modern ASR automatically applies **25-second chunking** with temporary WAV files to keep VRAM usage safe on consumer GPUs (≤12GB).

The chunking is transparent — you just pass the full audio file and receive a single concatenated result.

---

## Known Issues

**funasr 1.3.1 module import bugs**

Fun-ASR-Nano's model code uses absolute imports (`from ctc import CTC`, `from tools.utils import forced_align`) that fail in the packaged funasr distribution. Modern ASR automatically patches `sys.modules` before loading to inject the missing modules.

No user action required.

---

## Performance Notes

| Metric | Value |
|--------|-------|
| 7.5min audio on RTX 4070 Ti | ~47s |
| Chunk size | 25 seconds |
| Per-chunk latency | ~0.8–1.6s |
| VRAM usage | ~6–8GB |

- Weights downloaded from ModelScope (~2GB)
- Supports both `zh` and `en` with strong code-switching ability
