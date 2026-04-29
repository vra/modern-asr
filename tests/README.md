# Testing Guide

Modern ASR uses **pytest** for unit and integration testing. The test suite is designed to be fast, deterministic, and runnable in CI without downloading multi-gigabyte model weights.

---

## Quick Start

```bash
# Run all unit tests (fast, no network)
pytest tests/ --ignore=tests/test_integration.py -v

# Run everything including integration tests (downloads real weights)
pytest tests/ -v

# Run only integration tests
pytest tests/test_integration.py -v

# Skip slow tests
pytest tests/ -m "not slow" -v
```

---

## Test Structure

```
tests/
├── test_base.py              # ASRModel base class utilities
├── test_audio_llm.py         # AudioLLMModel generic logic
├── test_pipeline.py          # ASRPipeline input normalization & dispatch
├── test_utils_audio.py       # Audio loading, chunking, resampling
├── test_config.py            # BackendConfig, ModelConfig, PipelineConfig
├── test_registry.py          # Model registry (existing)
├── test_types.py             # Core types (existing)
├── test_models_whisper.py    # Whisper adapter (mocked)
├── test_models_sensevoice.py # SenseVoice adapter (mocked)
├── test_models_qwen.py       # Qwen3-ASR adapter (mocked)
├── test_models_funasr.py     # Fun-ASR adapters (mocked)
├── test_models_mimo.py       # MiMo adapter (mocked)
└── test_integration.py       # Real model loading (lightweight models only)
```

---

## Test Philosophy

### Unit Tests = Mocked External Dependencies

All model adapter tests use `unittest.mock` to avoid:
- Downloading weights from HuggingFace / ModelScope
- Loading large models into GPU/CPU memory
- Depending on specific package versions

Example pattern:

```python
@patch("funasr.AutoModel")
def test_transcribe(self, mock_automodel):
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"text": "你好"}]
    mock_automodel.return_value = mock_model

    pipe = ASRPipeline("sensevoice-small")
    result = pipe("audio.wav", language="zh")
    assert result.text == "你好"
```

### Integration Tests = Real Weights

The `test_integration.py` file loads actual models (only the smallest ones):

| Test | Model | Why |
|------|-------|-----|
| `test_load_and_transcribe_silence` | `whisper-tiny` (39M) | Smallest real model |
| `test_hot_swap` | `whisper-tiny` → `whisper-base` | Lifecycle verification |

Integration tests are marked with `@pytest.mark.integration` and `@pytest.mark.slow`.

---

## Writing New Tests

### Testing a New Model Adapter

If you add a model via `AudioLLMModel`, you typically only need to test:
1. **Configuration** — class attributes are set correctly
2. **Language mapping** — `_map_language()` works as expected
3. **Load mock** — `load()` calls the expected external API
4. **Transcribe mock** — `transcribe()` returns a valid `ASRResult`

Example for a new model `my-model`:

```python
# tests/test_models_my_model.py
from unittest.mock import MagicMock, patch
from modern_asr import ASRPipeline

class TestMyModel:
    @patch("my_package.MyModelClass")
    def test_load_and_transcribe(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = "hello"
        mock_cls.from_pretrained.return_value = mock_instance

        pipe = ASRPipeline("my-model")
        result = pipe("audio.wav", language="en")
        assert result.text == "hello"
```

### Testing Core Utilities

Tests for `ASRModel` utilities belong in `test_base.py`:

```python
def test_chunk_audio_splits_correctly():
    arr = np.ones(16000 * 10, dtype=np.float32)
    audio = AudioInput(data=arr, sample_rate=16000)
    chunks = chunk_audio(audio, chunk_duration=3.0)
    assert len(chunks) == 4
```

---

## Coverage

| Component | Tests | Coverage |
|-----------|-------|----------|
| `core/base.py` (ASRModel) | `test_base.py` | Device, dtype, audio I/O, chunking, lifecycle |
| `core/audio_llm.py` | `test_audio_llm.py` | Dynamic imports, language map, load, transcribe |
| `core/pipeline.py` | `test_pipeline.py` | Construction, switching, input normalization, tasks |
| `core/config.py` | `test_config.py` | Validation, YAML loading |
| `core/registry.py` | `test_registry.py` | Registration, lookup, creation |
| `core/types.py` | `test_types.py` | AudioInput, ASRResult, Segment |
| `utils/audio.py` | `test_utils_audio.py` | load_audio, chunk_audio, resample |
| `models/whisper_model.py` | `test_models_whisper.py` | Load, transcribe |
| `models/sensevoice.py` | `test_models_sensevoice.py` | Load, transcribe |
| `models/qwen_asr.py` | `test_models_qwen.py` | Load, transcribe, language map |
| `models/funasr_model.py` | `test_models_funasr.py` | Load, transcribe (Nano + Paraformer) |
| `models/mimo_asr.py` | `test_models_mimo.py` | Path resolution, language tags, transcribe |

---

## CI Recommendations

```yaml
# .github/workflows/test.yml (example)
- name: Run unit tests
  run: pytest tests/ --ignore=tests/test_integration.py -v

- name: Run integration tests
  run: pytest tests/test_integration.py -v
  if: github.ref == 'refs/heads/main'
```

Integration tests should only run on `main` branch or in scheduled jobs to save CI time and bandwidth.
