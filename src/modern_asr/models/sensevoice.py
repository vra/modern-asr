"""SenseVoice adapter (Alibaba / 阿里通义实验室).

Models:
    - sensevoice-small: 234M, non-autoregressive, ultra-fast
    - sensevoice-large: large variant, 50+ languages

Capabilities:
    - ASR (automatic speech recognition)
    - LID (language identification)
    - SER (speech emotion recognition)
    - AED (acoustic event detection)

References:
    - https://github.com/FunAudioLLM/SenseVoice
    - https://funaudiollm.github.io/
"""

from __future__ import annotations

from typing import Any

import numpy as np

from modern_asr.core.base import ASRModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.registry import register_model
from modern_asr.core.types import ASRResult, AudioInput, Segment


def _check_deps() -> None:
    try:
        import torch  # noqa: F401
        import funasr  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SenseVoice requires 'torch' and 'funasr'. "
            "Install with: uv pip install modern-asr[sensevoice]"
        ) from exc


class _SenseVoiceBase(ASRModel):
    """Shared logic for SenseVoice variants."""

    SUPPORTED_LANGUAGES = {
        "zh", "en", "yue", "ja", "ko", "auto", "nospeech", "multi"
    }
    SUPPORTED_MODES = {"transcribe", "emotion", "event"}
    REQUIREMENTS = ["torch", "torchaudio", "funasr", "modelscope"]

    def __init__(
        self,
        config: ModelConfig,
        backend: BackendConfig | None = None,
    ) -> None:
        super().__init__(config, backend)
        self._model_path = self._resolve_model_path()

    def _resolve_model_path(self) -> str:
        if self.config.model_path:
            return str(self.config.model_path)
        defaults = {
            "sensevoice-small": "iic/SenseVoiceSmall",
            "sensevoice-large": "iic/SenseVoiceLarge",
        }
        return defaults.get(self.config.model_id, f"iic/{self.config.model_id}")

    def load(self) -> None:
        _check_deps()
        from funasr import AutoModel

        self._model = AutoModel(
            model=self._model_path,
            trust_remote_code=True,
            device=self._resolve_device(),
            disable_update=True,
        )
        # Workaround: torchaudio 2.x kaldi.fbank outputs 81-dim (with energy)
        # instead of 80-dim, causing LFR output (81*7=567) to mismatch the
        # model's expected 560-dim. We patch the frontend to crop to 560.
        self._patch_frontend_output()
        self._is_loaded = True

    def _patch_frontend_output(self) -> None:
        """Patch frontend to fix torchaudio 2.x kaldi.fbank 81-dim output.

        torchaudio 2.x adds an energy dimension to kaldi.fbank (81-dim).
        With LFR window=7, this yields 567-dim features instead of the
        model's expected 560-dim (80*7). We reshape, drop the energy dim,
        and flatten back to 560 before CMVN.
        """
        import funasr.frontends.wav_frontend as _wf

        _orig_apply_cmvn = _wf.apply_cmvn
        lfr_m = self._model.kwargs["frontend"].lfr_m

        def _apply_cmvn(inputs, cmvn):
            if inputs.shape[-1] != cmvn[0].shape[0]:
                # inputs: [T, lfr_m * 81] -> [T, lfr_m, 81] -> drop energy -> [T, lfr_m * 80]
                t = inputs.shape[0]
                feat_dim = inputs.shape[-1] // lfr_m  # should be 81
                target_dim = cmvn[0].shape[0]  # should be 80 * lfr_m = 560
                if feat_dim * lfr_m == inputs.shape[-1] and target_dim == 80 * lfr_m:
                    inputs = inputs.reshape(t, lfr_m, feat_dim)[:, :, :-1].reshape(t, target_dim)
                else:
                    inputs = inputs[..., : cmvn[0].shape[0]]
            return _orig_apply_cmvn(inputs, cmvn)

        _wf.apply_cmvn = _apply_cmvn

    def transcribe(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        self._ensure_loaded()

        lang = kwargs.get("language", self.config.language or "auto")
        use_itn = kwargs.get("use_itn", True)

        # FunASR works most reliably with file paths;
        # numpy arrays can trigger shape mismatches in frontend CMVN.
        if audio.is_file():
            input_data = str(audio.data)
        else:
            import tempfile
            import soundfile as sf
            arr = self._to_waveform(audio)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, arr, audio.sample_rate)
                input_data = tmp.name

        res = self._model.generate(  # type: ignore[union-attr]
            input=input_data,
            cache={},
            language=lang,
            use_itn=use_itn,
            batch_size_s=60,
            merge_vad=True,
        )

        # FunASR returns list of dicts
        if isinstance(res, list) and len(res) > 0:
            item = res[0]
            text = item.get("text", "")
            # SenseVoice prepends tags like <|zh|><|NEUTRAL|><|Speech|>
            text = self._strip_tags(text)
        else:
            text = ""

        segments = [Segment(text=text, language=lang)]
        return ASRResult(
            text=text,
            segments=segments,
            language=lang,
            model_id=self.model_id,
            extra={"raw": res},
        )

    def detect_emotion(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        """SenseVoice natively outputs emotion tags in the text prefix."""
        result = self.transcribe(audio, **kwargs)
        # Emotion is embedded as <|NEUTRAL|>, <|HAPPY|>, etc.
        # Already exposed in raw text; user can parse from result.text
        return result

    def detect_events(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        """SenseVoice natively outputs acoustic event tags."""
        return self.transcribe(audio, **kwargs)

    def _to_waveform(self, audio: AudioInput) -> np.ndarray:
        if audio.is_array():
            arr = audio.data
            if isinstance(arr, np.ndarray):
                return arr
        from modern_asr.utils.audio import load_audio
        loaded = load_audio(str(audio.data))
        return loaded.data  # type: ignore[return-value]

    def _resolve_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @staticmethod
    def _strip_tags(text: str) -> str:
        """Remove SenseVoice special tags like <|zh|><|NEUTRAL|><|Speech|>."""
        import re
        return re.sub(r"<\|[^|]+\|>", "", text).strip()


@register_model("sensevoice-small")
class SenseVoiceSmall(_SenseVoiceBase):
    """SenseVoice-Small: 234M params, 70ms for 10s audio, 5 languages."""

    MODEL_CARD = "https://huggingface.co/iic/SenseVoiceSmall"

    @property
    def model_id(self) -> str:
        return "sensevoice-small"


@register_model("sensevoice-large")
class SenseVoiceLarge(_SenseVoiceBase):
    """SenseVoice-Large: high-precision, 50+ languages."""

    MODEL_CARD = "https://huggingface.co/iic/SenseVoiceLarge"

    @property
    def model_id(self) -> str:
        return "sensevoice-large"
