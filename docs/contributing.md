# Contributing

Thank you for your interest in Modern ASR! This project welcomes contributions of all kinds — bug fixes, new models, documentation improvements, and feature requests.

---

## Development Setup

```bash
git clone https://github.com/vra/modern-asr.git
cd modern-asr

# Create virtual environment and sync all dependencies
uv sync --all-extras

# The dev dependency group (pytest, ruff, mypy, pre-commit)
# is included by default via [tool.uv.default-groups].
```

---

## Adding a New Model

The easiest way to contribute is adding support for a new ASR model. Thanks to `AudioLLMModel`, this often takes less than 20 lines of code.

### Option A: Audio-LLM Model (Recommended)

If your model follows the standard pattern:

```python
# src/modern_asr/models/my_model.py
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

### Option B: Custom Adapter

If your model needs special handling (like MiMo or FireRedASR):

```python
from modern_asr.core.base import ASRModel
from modern_asr.core.registry import register_model

@register_model("my-custom-model")
class MyCustomModel(ASRModel):
    def load(self) -> None:
        ...

    def transcribe(self, audio, **kwargs):
        ...

    @property
    def model_id(self) -> str:
        return "my-custom-model"
```

---

## Code Style

We use **Ruff** for linting and formatting:

```bash
ruff check src/
ruff format src/
```

---

## Testing

```bash
pytest tests/
```

For model-specific tests, use optional markers:

```bash
pytest tests/ -m "not slow"
```

---

## Documentation

Docs are built with Material for MkDocs:

```bash
mkdocs serve    # Local preview
mkdocs build    # Build site/
```

When adding a new model, please also add:
1. A page under `docs/models/`
2. An entry in `mkdocs.yml` navigation
3. A row in `docs/models/index.md`

---

## Pull Request Checklist

- [ ] Code passes `ruff check` and `ruff format`
- [ ] Tests pass (or new tests added)
- [ ] Documentation updated
- [ ] Model added to comparison table
- [ ] `pyproject.toml` extras updated if new dependencies needed
