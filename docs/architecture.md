# Architecture

Modern ASR is designed around three layers: **Pipeline**, **Model Adapters**, and **Backends**. This separation lets you swap models, backends, and even entire frameworks without touching your application code.

---

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Application                           │
│         pipe = ASRPipeline("sensevoice-small")              │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      ASRPipeline                             │
│  • Input normalization (path → AudioInput)                  │
│  • Task dispatch (transcribe / translate / diarize)        │
│  • Model hot-swapping                                       │
│  • Chunked inference for long audio                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      ASRModel                                │
│  Abstract base class — every adapter implements this.       │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ AudioLLMModel│  │  FunASRModel │  │ WhisperModel     │  │
│  │ (generic)    │  │  (custom)    │  │ (custom)         │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      Backends                                │
│  • TransformersBackend   (HF transformers)                  │
│  • VLLMBackend           (vLLM inference engine)            │
│  • ONNXBackend           (ONNX Runtime)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Pipeline Layer

`ASRPipeline` is the only class most users interact with.

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("model-id")
result = pipe("audio.wav", language="zh")
```

Responsibilities:

- **Input normalization** — Converts file paths, bytes, or numpy arrays into a uniform `AudioInput`
- **Task dispatch** — Routes `task="transcribe"` to `model.transcribe()`, `task="translate"` to `model.translate()`, etc.
- **Model lifecycle** — Loads models on first use, supports hot-swapping via `switch_model()`, and auto-unloads with context managers
- **Configuration** — Reads `PipelineConfig` from YAML for persistent backend/model defaults

---

## 2. Model Adapter Layer

### ASRModel (Base Class)

Every model adapter subclasses `ASRModel` and implements three things:

```python
class MyModel(ASRModel):
    @property
    def model_id(self) -> str: ...

    def load(self) -> None: ...

    def transcribe(self, audio: AudioInput, **kwargs) -> ASRResult: ...
```

The base class provides a rich toolkit so adapters stay small:

| Method | Purpose |
|--------|---------|
| `_resolve_device()` | Map `"auto"` → `"cuda"` or `"cpu"` |
| `_resolve_dtype()` | Map `"float16"` / `"bfloat16"` → `torch.dtype` |
| `_to_waveform()` | Normalize any `AudioInput` to a 1-D numpy array |
| `_save_temp_audio()` | Write in-memory audio to a temporary WAV file |
| `_audio_to_file()` | Return a file path (creates temp file if needed) |
| `_chunk_audio()` | Split audio into fixed-duration chunks |
| `_chunked_transcribe()` | Generic chunk-then-merge for long audio |

### AudioLLMModel (Reusable Base)

Most modern ASR releases follow the same pattern:

```
audio → processor → model.generate() → processor.batch_decode() → text
```

`AudioLLMModel` captures this pattern. Subclasses configure behaviour through **class attributes** instead of writing boilerplate code.

```python
@register_model("my-audio-llm")
class MyAudioLLM(AudioLLMModel):
    HF_PATH = "org/MyAudioLLM-1B"
    PROCESSOR_CLS = "transformers.AutoProcessor"
    MODEL_CLS = "transformers.AutoModelForSpeechSeq2Seq"
    TRUST_REMOTE_CODE = True
    LANGUAGE_MAP = {"zh": "<|zh|>", "en": "<|en|>"}
    CHUNK_DURATION = 30.0
    DEFAULT_MAX_NEW_TOKENS = 256
```

That's it — 8 lines and the model is fully integrated.

#### Override Points

For models that deviate from the standard flow, override specific methods:

| Method | Override When |
|--------|---------------|
| `load()` | Custom loading (MiMo's `MimoAudio`, FireRedASR's official repo) |
| `_build_inputs()` | Special input format (Qwen-Omni's chat-template audio) |
| `_decode()` | Custom token decoding |
| `_build_result()` | Inject timestamps, speaker IDs, or confidence |
| `transcribe()` | Streaming, VAD-based splitting, or multi-turn context |

### Registry System

Models self-register via a decorator:

```python
from modern_asr.core.registry import register_model

@register_model("my-model")
class MyModel(ASRModel):
    ...
```

At import time, `auto_discover_models()` scans `modern_asr.models` and imports every module, triggering all decorators. This means **zero configuration files** are needed to add a new model — just drop a Python file in `src/modern_asr/models/`.

---

## 3. Backend Layer

The backend layer abstracts the inference framework. While most current adapters load models directly, the backend system is wired in and ready for migration.

### TransformersBackend

```python
from modern_asr.backends import TransformersBackend

backend = TransformersBackend(device="cuda", dtype="float16")
model, processor = backend.load("openai/whisper-small")
output_ids = backend.generate(inputs, max_new_tokens=256)
```

Features:
- Auto-detects processor type (`AutoProcessor` → `AutoTokenizer` fallback)
- Auto-detects model type (`AutoModelForSpeechSeq2Seq` → `AutoModel` fallback)
- Supports `trust_remote_code` for custom modeling files

### Future Backends

- **VLLMBackend** — For high-throughput batched inference
- **ONNXBackend** — For edge deployment (Moonshine already supports this)

---

## Adding a New Model (Complete Example)

Let's add a hypothetical model from HuggingFace.

### Step 1: Create the adapter file

`src/modern_asr/models/my_model.py`:

```python
"""MyModel adapter."""
from modern_asr.core.audio_llm import AudioLLMModel
from modern_asr.core.registry import register_model

@register_model("my-model-1b")
class MyModel1B(AudioLLMModel):
    HF_PATH = "org/MyModel-1B"
    PROCESSOR_CLS = "transformers.AutoProcessor"
    MODEL_CLS = "transformers.AutoModelForSpeechSeq2Seq"
    SUPPORTED_LANGUAGES = {"zh", "en", "ja"}
    CHUNK_DURATION = 30.0

    @property
    def model_id(self) -> str:
        return "my-model-1b"
```

### Step 2: Done

No step 2. The registry auto-discovers the module at runtime.

```python
from modern_asr import ASRPipeline
pipe = ASRPipeline("my-model-1b")
```

---

## Design Principles

1. **Local-first** — Every model downloads weights and runs on-device. No APIs.
2. **Minimal boilerplate** — New models need as little code as possible.
3. **Graceful degradation** — Missing dependencies don't crash the package; they only prevent loading that specific model.
4. **Framework agnostic** — The pipeline doesn't care if a model uses transformers, funasr, ONNX, or a custom C++ extension.
5. **Future-proof** — Dynamic imports and `trust_remote_code` mean tomorrow's architectures work today.
