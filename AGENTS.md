# Modern ASR — Agent Guide

## Project Overview

A unified, extensible Python toolkit for LLM-based Automatic Speech Recognition (ASR).
Designed with a plugin architecture so new models can be added with minimal boilerplate.

## Directory Structure

```
src/modern_asr/
├── __init__.py              # Public API exports, triggers model auto-discovery
├── __main__.py              # python -m modern_asr entrypoint
├── cli.py                   # CLI: modern-asr list / transcribe
├── core/
│   ├── base.py              # ASRModel abstract base class
│   ├── audio_llm.py         # AudioLLMModel reusable base for audio-encoder + LLM-decoder
│   ├── subprocess_mixin.py  # SubprocessIsolatedMixin for dependency-conflicted models
│   ├── types.py             # AudioInput, ASRResult, Segment, WordTimestamp
│   ├── config.py            # ModelConfig, BackendConfig, PipelineConfig (Pydantic)
│   ├── registry.py          # @register_model decorator, create_model(), list_models()
│   └── pipeline.py          # ASRPipeline high-level API
├── backends/
│   ├── base.py              # InferenceBackend ABC
│   ├── subprocess_backend.py # Generic JSON-over-stdio subprocess worker backend
│   ├── transformers_backend.py
│   ├── vllm_backend.py
│   └── onnx_backend.py
├── models/
│   ├── __init__.py          # auto_discover_models() trigger
│   ├── fireredasr.py        # Xiaohongshu
│   ├── sensevoice.py        # Alibaba
│   ├── qwen_asr.py          # Qwen3-ASR
│   ├── qwen_omni.py         # Qwen2.5-Omni
│   ├── funasr_model.py      # Fun-ASR / Paraformer
│   ├── mimo_asr.py          # Xiaomi
│   ├── midasheng.py         # Xiaomi audio understanding
│   ├── canary_qwen.py       # NVIDIA
│   ├── glm_asr.py           # Zhipu AI
│   ├── granite_speech.py    # IBM
│   ├── whisper_model.py     # OpenAI
│   ├── moonshine.py         # Useful Sensors
│   └── llama_omni.py        # LLaMA-Omni
└── utils/
    ├── audio.py             # load_audio(), chunk_audio()
    ├── auto_install.py      # Auto-install deps, git repos, and HF weights on first use
    └── log.py               # Unified RichHandler logging with ISO timestamps
```

## Subprocess Isolation (Dependency Conflicts)

When an upstream model pins a dependency version that conflicts with the main
environment (e.g. ``transformers==4.57.6`` vs ``transformers>=5.7.0``), use
``SubprocessIsolatedMixin`` to transparently spawn an isolated worker.

Architecture:
- ``SubprocessBackend`` — generic NDJSON-over-stdin/stdout communication layer.
- ``scripts/subprocess_worker.py`` — generic worker; loads *any* registered model.
- ``SubprocessIsolatedMixin`` — reusable mixin; subclasses only set class attrs.

### How to add an isolated model

```python
from modern_asr.core.subprocess_mixin import SubprocessIsolatedMixin

@register_model("my-model")
class MyModel(SubprocessIsolatedMixin, AudioLLMModel):
    SUBPROCESS_VENV = ".venv_my_model"
    SUBPROCESS_ENV_VAR = "MODERN_ASR_MY_MODEL_VENV"
    SUBPROCESS_CHECK = staticmethod(
        lambda: int(transformers.__version__.split(".")[0]) >= 5
    )

    def load(self):
        self._try_native_then_subprocess(native_load=super().load)

    def transcribe(self, audio, **kwargs):
        self._ensure_loaded()
        audio_path = self._audio_to_file(audio)
        if getattr(self, "_subprocess_backend", None) is not None:
            return self._subprocess_transcribe(audio_path, **kwargs)
        return super().transcribe(audio, **kwargs)
```

### Venv discovery order

1. ``$SUBPROCESS_ENV_VAR`` (if set)
2. ``{project_root}/{SUBPROCESS_VENV}`` (if set)
3. ``{project_root}/.venv_{model_slug}`` (fallback convention)

### Creating an isolated venv

```bash
cd /home/ws/ws/projects/modern-asr
python3.10 -m venv .venv_my_model
source .venv_my_model/bin/activate
pip install transformers==4.57.6 accelerate torch <model-specific-packages>
```

## Adding a New Model

1. Create `src/modern_asr/models/<vendor>_<model>.py`
2. Subclass `ASRModel`
3. Implement `model_id`, `load()`, `transcribe()`
4. Decorate with `@register_model("my-model-id")`
5. Add optional dependency group in `pyproject.toml`

Template:

```python
from modern_asr import ASRModel, register_model, ModelConfig
from modern_asr.core.types import ASRResult, AudioInput

@register_model("my-model")
class MyModel(ASRModel):
    SUPPORTED_LANGUAGES = {"zh", "en"}
    SUPPORTED_MODES = {"transcribe"}

    @property
    def model_id(self) -> str:
        return "my-model"

    def load(self) -> None:
        ...
        self._is_loaded = True

    def transcribe(self, audio: AudioInput, **kwargs) -> ASRResult:
        ...
        return ASRResult(text=..., model_id=self.model_id)
```

## Code Style

- Use `from __future__ import annotations` in every module
- Type hints for all public methods
- Ruff for linting, mypy for type checking
- `rich` for CLI pretty-printing

## Testing

```bash
cd /home/ws/ws/projects/modern-asr
PYTHONPATH=src pytest tests/ -v
```

### End-to-End Model Testing Notes

Use **Python 3.10** (`/home/ws/miniforge3/envs/py310/bin/python`) for all model tests.
The py39 env lacks compatibility with newer `torchvision` and some models use py3.10+
syntax (`int | list` union types).

**Verified working (CPU, py310):**
- `whisper-tiny/base/small/medium/large-v3/v3-turbo`
- `moonshine-tiny`
- `funasr-nano`
- `paraformer-zh`, `paraformer-large`
- `sensevoice-small`
- `glm-asr-nano-2512`
- `fireredasr-aed`
- `granite-speech-3.3-8b` (8B, slow on CPU)

**Fixed during testing:**
- `granite-speech-3.3-8b`: Requires `<|audio|>` chat template, torch.Tensor audio input,
  and `device=` param (not `sampling_rate=`) in processor call.
- `fireredasr-llm`: Fixed `__init__` order bug (`_asr_type` must be set before
  `_resolve_model_dir`). Requires Qwen2-7B-Instruct base LLM symlink. Patched
  tokenizer for transformers 5.x (`apply_chat_template` returns `BatchEncoding`).
- `whisper-*`: `language="auto"` must be converted to `None` for OpenAI whisper.
- `ensure_hf()`: Updated to detect incomplete downloads by checking for weight files
  (`.safetensors`/`.bin`/`.pt`) instead of just non-empty directories.
- `torchvision`: Must use version matching torch (e.g. `torchvision==0.21.0+cu124`
  for `torch==2.6.0+cu124`). Incompatible versions cause `torchvision::nms` error
  which breaks `transformers` imports.
- `mimo-asr-v2.5`: Patched `flash_attn` fallback. Added `device` pass-through.
  Code uses Python 3.10 `|` union syntax.

**Known limitations:**
- `qwen3-asr-*`: Resolved via subprocess isolation in `.venv_qwen310` with
  `transformers==4.57.6`. Falls back transparently when the main env has
  `transformers>=5.x`.
- `glm-asr-nano-2512`: Resolved via subprocess isolation in `.venv_glm` with
  `transformers` installed from source (`git+https://github.com/huggingface/transformers`).
  The release versions (including 5.7.0) do not yet include the `glmasr` architecture.
- `midashenglm-7b`: Incomplete downloads (large files, slow mirror).

**Environment for testing:**
```bash
conda activate py310
export MODERN_ASR_CACHE_DIR=/mnt/hdd/.cache/modern-asr
export HF_HOME=/mnt/hdd/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
```

## Build / Package

```bash
uv sync --all-extras
```
