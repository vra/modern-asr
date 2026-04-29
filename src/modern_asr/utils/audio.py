"""Audio loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modern_asr.core.types import AudioInput


def load_audio(
    path: str | Path,
    target_sr: int = 16000,
    mono: bool = True,
    dtype: str = "float32",
) -> AudioInput:
    """Load an audio file and normalize to the target format.

    This utility tries multiple backends in order of preference:
    ``torchaudio``, ``librosa``, ``soundfile``, and falls back to
    ``wave`` + ``array`` for raw WAV if nothing else is available.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    arr, sr = _load_with_best_backend(str(path), target_sr, mono)
    return AudioInput(
        data=arr.astype(dtype),
        sample_rate=sr,
        channels=1 if mono else (arr.ndim if arr.ndim > 1 else 1),
        dtype=dtype,
        source=str(path),
    )


def _load_with_best_backend(
    path: str,
    target_sr: int,
    mono: bool,
) -> tuple[np.ndarray, int]:
    """Attempt loading with available libraries."""
    # Prefer torchaudio
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if mono and waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
            sr = target_sr
        return waveform.numpy(), sr
    except Exception:
        pass

    # Fallback to librosa
    try:
        import librosa
        arr, sr = librosa.load(path, sr=target_sr, mono=mono)
        return arr, sr
    except Exception:
        pass

    # Fallback to soundfile
    try:
        import soundfile as sf
        arr, sr = sf.read(path, dtype="float32")
        if mono and arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != target_sr:
            # Simple linear interpolation resampling (not ideal but works)
            arr = _resample_simple(arr, sr, target_sr)
            sr = target_sr
        return arr, sr
    except Exception:
        pass

    # Last resort: built-in wave
    import wave
    import struct
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n_channels = w.getnchannels()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
        fmt = f"{n_frames * n_channels}h"
        samples = struct.unpack(fmt, raw)
        arr = np.array(samples, dtype=np.float32) / 32768.0
        if n_channels > 1 and mono:
            arr = arr.reshape(-1, n_channels).mean(axis=1)
        if sr != target_sr:
            arr = _resample_simple(arr, sr, target_sr)
            sr = target_sr
        return arr, sr


def _resample_simple(arr: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple 1D linear resampling."""
    if orig_sr == target_sr:
        return arr
    ratio = target_sr / orig_sr
    n = int(len(arr) * ratio)
    idx = np.linspace(0, len(arr) - 1, n)
    idx_l = np.floor(idx).astype(int)
    idx_r = np.minimum(idx_l + 1, len(arr) - 1)
    frac = idx - idx_l
    return arr[idx_l] * (1 - frac) + arr[idx_r] * frac


def chunk_audio(
    audio: AudioInput,
    chunk_duration: float = 30.0,
    overlap: float = 0.0,
) -> list[AudioInput]:
    """Split audio into fixed-duration chunks with optional overlap.

    Args:
        audio: Input audio.
        chunk_duration: Chunk length in seconds.
        overlap: Overlap between consecutive chunks in seconds.

    Returns:
        List of `AudioInput` chunks.
    """
    if not audio.is_array():
        raise ValueError("chunk_audio requires an array-based AudioInput")

    arr = audio.data
    sr = audio.sample_rate
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    step = chunk_samples - overlap_samples

    chunks = []
    for start in range(0, len(arr), step):
        end = min(start + chunk_samples, len(arr))
        chunk_arr = arr[start:end]
        chunks.append(
            AudioInput(
                data=chunk_arr,
                sample_rate=sr,
                channels=audio.channels,
                dtype=audio.dtype,
                source=f"{audio.source or 'chunk'}_{start}_{end}",
            )
        )
        if end >= len(arr):
            break
    return chunks
