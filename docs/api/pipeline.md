# ASRPipeline

`ASRPipeline` is the high-level user-facing API for Modern ASR. It handles model loading, input normalization, task dispatch, and result formatting.

---

## Constructor

```python
ASRPipeline(
    model_id: str | None = None,
    model_config: ModelConfig | None = None,
    backend: BackendConfig | None = None,
    config_path: str | Path | None = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str \| None` | Initial model to load. If `None`, call `switch_model()` later. |
| `model_config` | `ModelConfig \| None` | Override configuration for the model. |
| `backend` | `BackendConfig \| None` | Global backend settings (device, dtype, etc.). |
| `config_path` | `str \| Path \| None` | Path to a YAML pipeline configuration file. |

---

## Calling the Pipeline

```python
result = pipe(
    audio: str | Path | AudioInput | np.ndarray,
    task: str = "transcribe",
    language: str | None = None,
    **kwargs,
) -> ASRResult
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio` | `str \| Path \| AudioInput \| np.ndarray` | Audio input. File paths are auto-loaded. |
| `task` | `str` | Task to perform: `"transcribe"`, `"translate"`, `"diarize"`, `"emotion"`, `"event"`. |
| `language` | `str \| None` | Target language code (model-specific mapping applied automatically). |
| `**kwargs` | — | Forwarded to the model's inference method. |

---

## Methods

### `switch_model(model_id: str) -> None`

Hot-swap to a different model. The previous model is unloaded automatically.

```python
pipe = ASRPipeline("sensevoice-small")
pipe.switch_model("qwen3-asr-0.6b")
```

### `unload() -> None`

Explicitly unload the current model and free GPU memory.

```python
pipe.unload()
```

### Context Manager

```python
with ASRPipeline("whisper-large-v3") as pipe:
    result = pipe("audio.wav")
# Auto-unloaded on exit
```

---

## Configuration YAML

Create a `pipeline.yaml` for persistent defaults:

```yaml
default_backend:
  name: transformers
  device: cuda
  dtype: float16

models:
  sensevoice-small:
    language: zh
    return_timestamps: false

  whisper-small:
    language: en
    beam_size: 5
```

Load it:

```python
pipe = ASRPipeline(config_path="pipeline.yaml")
```
