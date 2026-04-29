# Model Overview

Modern ASR supports **23 models** across **14 families**, ranging from 39M-parameter edge models to 8B-parameter Audio LLMs. Every model runs **locally** — no API keys, no network after download.

---

## Comparison Table

| Model | Params | Speed | Chinese | English | Dialects | Best For |
|-------|--------|-------|---------|---------|----------|----------|
| [Whisper Tiny](whisper.md) | 39M | ⚡⚡⚡ | ★★☆ | ★★★ | — | Fast prototyping, edge |
| [Whisper Base](whisper.md) | 74M | ⚡⚡⚡ | ★★☆ | ★★★ | — | Balanced speed/quality |
| [Whisper Small](whisper.md) | 244M | ⚡⚡ | ★★★ | ★★★ | — | Good quality general ASR |
| [Whisper Medium](whisper.md) | 769M | ⚡⚡ | ★★★ | ★★★ | — | High-quality general ASR |
| [Whisper Large v3](whisper.md) | 1.5B | ⚡ | ★★★ | ★★★ | — | Best Whisper quality |
| [Whisper Large v3 Turbo](whisper.md) | 809M | ⚡⚡ | ★★★ | ★★★ | — | Large quality, faster |
| [SenseVoice Small](sensevoice.md) | — | ⚡⚡⚡ | ★★★ | ★★★ | 粤语 | Chinese-centric, emotion |
| [SenseVoice Large](sensevoice.md) | — | ⚡⚡ | ★★★ | ★★★ | 粤语 | Best SenseVoice quality |
| [FireRedASR AED](fireredasr.md) | — | ⚡ | ★★★ | ★★☆ | 粤语 | Named entities, accuracy |
| [FireRedASR LLM](fireredasr.md) | — | ⚡ | ★★★ | ★★☆ | 粤语 | LLM-based Chinese ASR |
| [Qwen3-ASR 0.6B](qwen-asr.md) | 0.6B | ⚡⚡⚡ | ★★★ | ★★★ | 22 dialects | Speed, dialects |
| [Qwen3-ASR 1.7B](qwen-asr.md) | 1.7B | ⚡⚡ | ★★★ | ★★★ | 22 dialects | Higher accuracy |
| [Qwen2.5-Omni 7B](qwen-omni.md) | 7B | ⚡ | ★★★ | ★★★ | 粤语 | Multimodal (audio+text+image) |
| [Fun-ASR-Nano](funasr-nano.md) | 0.8B | ⚡⚡⚡ | ★★★ | ★★★ | — | Massive training data |
| [Paraformer Zh](paraformer.md) | 220M | ⚡⚡⚡ | ★★★ | ★★☆ | — | Streaming Chinese ASR |
| [Paraformer Large](paraformer.md) | — | ⚡⚡ | ★★★ | ★★☆ | — | VAD + punctuation |
| [MiMo-V2.5-ASR](mimo-asr.md) | 8B | ⚡ | ★★★ | ★★★ | 吴/粤/闽/川 | Dialects, noise, songs |
| [GLM-ASR Nano](glm-asr.md) | 1.5B | ⚡⚡ | ★★★ | ★★★ | — | Chinese, English |
| [Canary-Qwen 2.5B](canary-qwen.md) | 2.5B | ⚡⚡ | ★★★ | ★★★ | — | NVIDIA optimized |
| [Granite-Speech 3.3-8B](granite-speech.md) | 8B | ⚡ | ★★★ | ★★★ | — | IBM enterprise |
| [LLaMA-Omni 8B](llama-omni.md) | 8B | ⚡ | ★★☆ | ★★★ | — | Speech-to-text chat |
| [MiDashengLM 7B](midasheng.md) | 7B | ⚡ | ★★★ | ★★★ | — | General audio understanding |
| [Moonshine Tiny](moonshine.md) | — | ⚡⚡⚡ | ★★☆ | ★★★ | — | Edge ONNX deployment |

**Legend:** ⚡⚡⚡ = Fastest, ★★★ = Best quality

---

## By Use Case

### Best for Chinese (Mandarin)
- **Speed**: SenseVoice Small, Qwen3-ASR 0.6B, Fun-ASR-Nano
- **Accuracy**: FireRedASR LLM, MiMo-V2.5-ASR, Qwen3-ASR 1.7B
- **Dialects**: MiMo-V2.5-ASR (Wu, Yue, Minnan, Sichuan), Qwen3-ASR (22 dialects)

### Best for English
- **Speed**: Whisper Tiny/Base, Moonshine Tiny
- **Accuracy**: Whisper Large v3, Granite-Speech 3.3-8B

### Best for Code-Switching (Zh/En mix)
- MiMo-V2.5-ASR, Qwen3-ASR, MiDashengLM-7B

### Best for Noisy Environments
- MiMo-V2.5-ASR, SenseVoice Large, Qwen3-ASR 1.7B

### Best for Edge / Low Resource
- Moonshine Tiny (ONNX), Whisper Tiny, Paraformer Zh

### Best for Multi-Speaker / Meetings
- MiMo-V2.5-ASR (native diarization support)

---

## Language Code Reference

Different model families use different language code styles:

| Code | Meaning | Models |
|------|---------|--------|
| `zh` | Mandarin Chinese | All |
| `en` | English | All |
| `yue` | Cantonese (粤语) | SenseVoice, FireRedASR, Qwen, MiMo |
| `wuu` | Wu/Shanghainese (吴语) | Qwen3-ASR, MiMo |
| `nan` | Hokkien/Southern Min (闽南语) | Qwen3-ASR, MiMo |
| `cmn` | Mandarin (alternative code) | Qwen3-ASR, MiMo |
| `ja` | Japanese | Whisper, SenseVoice, Qwen |
| `ko` | Korean | Whisper, SenseVoice, Qwen |
| `auto` | Auto-detect | Most |
| `multi` | Multilingual mode | Most |

Modern ASR maps these codes automatically where possible. If you pass an unsupported code, the model will either auto-detect or raise an error.
