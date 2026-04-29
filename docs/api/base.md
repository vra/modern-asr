# ASRModel (Base Class)

`ASRModel` is the abstract base class that every model adapter must subclass. It defines the contract between the pipeline and individual models, and provides a rich set of shared utilities.

---

## Abstract Methods

### `model_id` (property)

```python
@property
def model_id(self) -> str: ...
```

Return the canonical model identifier string, e.g. `"sensevoice-small"`.

### `load()`

```python
def load(self) -> None: ...
```

Load model weights and processor into memory. Must set `self._is_loaded = True` on success.

### `transcribe()`

```python
def transcribe(self, audio: AudioInput, **kwargs: Any) -> ASRResult: ...
```

Run speech recognition on `audio` and return an `ASRResult`.

---

## Optional Task Methods

### `translate()`

Default implementation falls back to `transcribe` with `task="translate"`.

### `diarize()`

Default raises `NotImplementedError`. Override if the model supports speaker diarization.

### `detect_emotion()`

Default raises `NotImplementedError`. Override if the model supports emotion detection.

### `detect_events()`

Default raises `NotImplementedError`. Override if the model supports acoustic event detection.

---

## Shared Utilities

### Device & Precision

```python
def _resolve_device(self, device: str | None = None) -> str
```
Map `"auto"` → `"cuda"` or `"cpu"`. Uses `self.backend.device` if no override given.

```python
def _resolve_dtype(self, dtype: str | None = None) -> Any
```
Map `"float16"` / `"bfloat16"` / `"float32"` → `torch.dtype`.

### Audio I/O

```python
def _to_waveform(self, audio: AudioInput) -> np.ndarray
```
Extract a 1-D numpy waveform from any `AudioInput` (array, file path, or bytes).

```python
def _save_temp_audio(self, audio: AudioInput, suffix: str = ".wav") -> str
```
Write in-memory audio to a temporary WAV file. Returns the absolute path.

```python
def _audio_to_file(self, audio: AudioInput) -> str
```
Return a file path. If `audio` is already a file, returns its path. If it's an array, writes a temp file.

### Chunking

```python
def _chunk_audio(
    self,
    audio: AudioInput,
    chunk_duration: float | None = None,
    overlap: float = 0.0,
) -> list[AudioInput]
```
Split audio into fixed-duration chunks. Uses `self.CHUNK_DURATION` as default.

```python
def _chunked_transcribe(
    self,
    audio: AudioInput,
    chunk_duration: float | None = None,
    overlap: float = 0.0,
    **kwargs: Any,
) -> ASRResult
```
Generic long-audio handler: chunk → transcribe each → concatenate results.

---

## Class Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `MODEL_CARD` | `str` | URL to model card / documentation |
| `SUPPORTED_LANGUAGES` | `set[str]` | Language codes this model supports |
| `SUPPORTED_MODES` | `set[str]` | Tasks supported (`transcribe`, `translate`, `diarize`, etc.) |
| `REQUIREMENTS` | `list[str]` | Python package names required by this model |
| `CHUNK_DURATION` | `float` | Seconds per chunk for automatic splitting (0.0 = disabled) |
