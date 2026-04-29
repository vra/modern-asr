# AudioLLMModel

`AudioLLMModel` is a reusable base class for the increasingly common **"audio encoder / adapter + LLM decoder"** architecture used by models such as GLM-ASR, Qwen2.5-Omni, and many future releases.

Subclasses configure behaviour through **class attributes** and only need to implement `model_id`. Everything else has sensible defaults.

---

## Minimal Example

```python
from modern_asr.core.audio_llm import AudioLLMModel
from modern_asr.core.registry import register_model

@register_model("my-audio-llm")
class MyAudioLLM(AudioLLMModel):
    HF_PATH = "org/MyAudioLLM-1B"
    PROCESSOR_CLS = "transformers.AutoProcessor"
    MODEL_CLS = "transformers.AutoModelForSpeechSeq2Seq"
    LANGUAGE_MAP = {"zh": "<|zh|>", "en": "<|en|>"}
    CHUNK_DURATION = 30.0

    @property
    def model_id(self) -> str:
        return "my-audio-llm"
```

---

## Class Attributes

| Attribute | Default | Description |
|-----------|---------|-------------|
| `HF_PATH` | `""` | Default HuggingFace hub ID or local path |
| `PROCESSOR_CLS` | `"transformers.AutoProcessor"` | Dotted import path for processor/tokenizer |
| `MODEL_CLS` | `"transformers.AutoModelForSpeechSeq2Seq"` | Dotted import path for model class |
| `TRUST_REMOTE_CODE` | `True` | Passed to `from_pretrained` |
| `LOW_CPU_MEM_USAGE` | `True` | Passed to `from_pretrained` for model |
| `LANGUAGE_MAP` | `{}` | Map short codes → model-specific tags |
| `DEFAULT_MAX_NEW_TOKENS` | `256` | Fallback `max_new_tokens` |
| `DEFAULT_GENERATION_KWARGS` | `{}` | Extra kwargs for `model.generate()` |

### Common `PROCESSOR_CLS` Values

- `transformers.AutoProcessor`
- `transformers.AutoTokenizer`
- `transformers.Qwen2_5OmniProcessor`

### Common `MODEL_CLS` Values

- `transformers.AutoModelForSpeechSeq2Seq`
- `transformers.AutoModelForCausalLM`
- `transformers.AutoModel`
- `transformers.Qwen2_5OmniModel`
- Custom paths like `qwen_asr.core.modeling_qwen3_asr.Qwen3ASRForConditionalGeneration`

---

## Methods

### `load()`

Default implementation:
1. Resolves device and dtype via `_resolve_device()` / `_resolve_dtype()`
2. Dynamically imports `PROCESSOR_CLS` and `MODEL_CLS`
3. Calls `processor_cls.from_pretrained()` and `model_cls.from_pretrained()`
4. Sets `self._is_loaded = True`

Override if your model needs custom loading (e.g. MiMo's `MimoAudio` class).

### `transcribe(audio, **kwargs)`

Default implementation:
1. If `CHUNK_DURATION > 0` and audio exceeds chunk size → uses `_chunked_transcribe()`
2. Otherwise calls `_transcribe_single(audio, **kwargs)`

Override for streaming, VAD-based splitting, or multi-turn context.

### `_transcribe_single(audio, **kwargs)`

Runs one forward pass:
1. `_build_inputs(audio)` → model inputs
2. `model.generate()` with merged generation kwargs
3. `_decode(generated_ids)` → text
4. `_build_result(text)` → `ASRResult`

### `_build_inputs(audio, **kwargs)`

Default: `processor(waveform, sampling_rate=sr, return_tensors="pt")`

Override for:
- Chat-template prompting (Qwen-Omni)
- Discrete token inputs (MiMo)
- Multi-turn context

### `_decode(generated_ids, **kwargs)`

Default: `processor.batch_decode(ids, skip_special_tokens=True)[0]`

Override for custom special-token handling.

### `_build_result(text, **kwargs)`

Default: `ASRResult(text=text.strip(), segments=[...], language=..., model_id=...)`

Override to inject word-level timestamps, speaker IDs, or confidence scores.

### `_map_language(lang)`

Map a short language code to the model-specific tag using `LANGUAGE_MAP`.

---

## Override Pattern Matrix

| If your model needs… | Override |
|----------------------|----------|
| Custom loading (cloned repo, special tokenizer) | `load()` |
| Chat-template or prompt-based audio input | `_build_inputs()` |
| Discrete audio tokens instead of raw waveform | `_build_inputs()` |
| Streaming / VAD-based chunking | `transcribe()` |
| Custom decode (special tokens) | `_decode()` |
| Timestamps / speaker IDs in result | `_build_result()` |
| Different language tag format | `LANGUAGE_MAP` |
