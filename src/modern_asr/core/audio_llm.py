"""Generic base class for Audio-LLM style ASR models.

This module provides ``AudioLLMModel`` — a reusable abstraction for the
increasingly common "audio encoder / adapter + LLM decoder" architecture
used by models such as GLM-ASR, Qwen2.5-Omni, MiMo-V2.5-ASR, and many
future releases.

Design goals
------------
* **Minimal boilerplate** for new models — often just a few class attrs.
* **Pluggable loading** — works with ``transformers`` Auto classes,
  custom modelling files (``trust_remote_code``), or subclass overrides.
* **Automatic chunking** — long audio is split transparently when
  ``CHUNK_DURATION > 0``.
* **Future-proof** — new Audio-LLM architectures only need to override
  ``_build_inputs`` and/or ``_decode`` if they deviate from the standard
  ``processor(audio) -> model.generate() -> processor.batch_decode()`` flow.

Example::

    @register_model("my-audio-llm")
    class MyAudioLLM(AudioLLMModel):
        HF_PATH = "org/MyAudioLLM-1B"
        MODEL_CLS = "transformers.AutoModelForSpeechSeq2Seq"
        PROCESSOR_CLS = "transformers.AutoProcessor"
        LANGUAGE_MAP = {"zh": "<|zh|>", "en": "<|en|>"}
        CHUNK_DURATION = 30.0
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import numpy as np

from modern_asr.core.base import ASRModel

if TYPE_CHECKING:
    from modern_asr.core.config import BackendConfig, ModelConfig
    from modern_asr.core.types import ASRResult, AudioInput


class AudioLLMModel(ASRModel):
    """Reusable base for Audio-LLM ASR models.

    Subclasses configure behaviour through **class attributes** and only
    need to implement ``model_id`` (property).  Everything else has
    sensible defaults that cover the majority of today's Audio-LLM
    releases.
    """

    # --- Class-level configuration --------------------------------------

    HF_PATH: str = ""
    """Default HuggingFace hub ID (or local path)."""

    PROCESSOR_CLS: str = "transformers.AutoProcessor"
    """Dotted import path for the processor/tokenizer class.

    Common choices:
    * ``"transformers.AutoProcessor"``
    * ``"transformers.AutoTokenizer"``
    * ``"transformers.Qwen2_5OmniProcessor"`` (for explicit classes)
    """

    MODEL_CLS: str = "transformers.AutoModelForSpeechSeq2Seq"
    """Dotted import path for the model class.

    Common choices:
    * ``"transformers.AutoModelForSpeechSeq2Seq"``
    * ``"transformers.AutoModelForCausalLM"``
    * ``"transformers.AutoModel"``
    * A custom path such as ``"qwen_asr.core.transformers_backend.modeling_qwen3_asr.Qwen3ASRForConditionalGeneration"``
    """

    TRUST_REMOTE_CODE: bool = True
    """Passed to ``from_pretrained`` for both processor and model."""

    LOW_CPU_MEM_USAGE: bool = True
    """Passed to ``from_pretrained`` for the model."""

    LANGUAGE_MAP: dict[str, str | None] = {}
    """Map from short language codes (``zh``, ``en`` …) to model-specific
    tokens or strings.  A value of ``None`` means "do not inject a tag"."""

    DEFAULT_MAX_NEW_TOKENS: int = 256
    """Fallback ``max_new_tokens`` when not provided by user or config."""

    DEFAULT_GENERATION_KWARGS: dict[str, Any] = {}
    """Extra kwargs forwarded to ``model.generate()`` — subclass defaults."""

    # CHUNK_DURATION is inherited from ASRModel (0.0 disables automatic chunking)

    # --- Construction ---------------------------------------------------

    def __init__(
        self,
        config: ModelConfig,
        backend: BackendConfig | None = None,
    ) -> None:
        super().__init__(config, backend)
        self._hf_path = self._resolve_hf_path()

    def _resolve_hf_path(self) -> str:
        if self.config.model_path:
            return str(self.config.model_path)
        return self.HF_PATH

    def _auto_install_requirements(self) -> None:
        """Auto-install missing packages declared in ``REQUIREMENTS``."""
        from modern_asr.utils.auto_install import ensure_pypi

        for spec in getattr(self, "REQUIREMENTS", []):
            if spec:
                ensure_pypi(spec)

    # --- Loading --------------------------------------------------------

    def load(self) -> None:
        """Load processor and model using the configured class paths."""
        from modern_asr.utils.log import get_logger

        logger = get_logger(__name__)
        logger.info("Loading %s from %s", self.model_id, self._hf_path)

        self._auto_install_requirements()
        device = self._resolve_device()
        dtype = self._resolve_dtype()

        processor_cls = self._import_cls(self.PROCESSOR_CLS)
        model_cls = self._import_cls(self.MODEL_CLS)

        self._processor = processor_cls.from_pretrained(
            self._hf_path,
            trust_remote_code=self.TRUST_REMOTE_CODE,
        )

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.TRUST_REMOTE_CODE,
            "low_cpu_mem_usage": self.LOW_CPU_MEM_USAGE,
        }
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        if device != "cpu":
            load_kwargs["device_map"] = device
        else:
            load_kwargs["device_map"] = None

        self._model = model_cls.from_pretrained(self._hf_path, **load_kwargs)
        self._is_loaded = True
        logger.info("%s loaded on %s (dtype=%s)", self.model_id, device, dtype)

    # --- Inference ------------------------------------------------------

    def transcribe(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        """Transcribe ``audio``.

        If ``self.CHUNK_DURATION > 0`` and the audio duration exceeds the
        chunk size, the generic ``_chunked_transcribe`` fallback is used.
        Subclasses may override this method for smarter strategies
        (VAD-based splitting, streaming, etc.).
        """
        self._ensure_loaded()

        # Fast path: no automatic chunking configured
        if self.CHUNK_DURATION <= 0:
            return self._transcribe_single(audio, **kwargs)

        # Determine audio duration
        waveform = self._to_waveform(audio)
        duration = len(waveform) / audio.sample_rate

        if duration <= self.CHUNK_DURATION:
            return self._transcribe_single(audio, **kwargs)

        return self._chunked_transcribe(
            audio,
            chunk_duration=self.CHUNK_DURATION,
            overlap=0.0,
            **kwargs,
        )

    def _transcribe_single(self, audio: AudioInput, **kwargs: Any) -> ASRResult:
        """Run a single forward pass on ``audio``.

        Subclasses with non-standard input formats (e.g. MiMo's discrete
        audio tokens) should override this method.
        """
        import torch

        inputs = self._build_inputs(audio, **kwargs)
        inputs = {
            k: v.to(self._model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        gen_kwargs = {
            "max_new_tokens": kwargs.get(
                "max_new_tokens",
                self.config.max_new_tokens or self.DEFAULT_MAX_NEW_TOKENS,
            ),
            "do_sample": False,
        }
        # Merge subclass defaults + per-call overrides + config beam_size
        gen_kwargs.update(self.DEFAULT_GENERATION_KWARGS)
        gen_kwargs.update(
            {k: v for k, v in kwargs.items() if k not in ("language", "task")}
        )
        if self.config.beam_size is not None:
            gen_kwargs.setdefault("num_beams", self.config.beam_size)

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **gen_kwargs)

        text = self._decode(generated_ids, **kwargs)
        return self._build_result(text, **kwargs)

    # --- Input / output builders (override points) ----------------------

    def _build_inputs(self, audio: AudioInput, **kwargs: Any) -> dict[str, Any]:
        """Convert ``audio`` into the dict expected by ``model.generate()``.

        The default implementation uses the processor to process a raw
        waveform::

            processor(waveform, sampling_rate=sr, return_tensors="pt")

        Subclasses may override for:
        * Chat-template style prompting (Qwen-Omni)
        * Discrete token inputs (MiMo)
        * Multi-turn context
        """
        waveform = self._to_waveform(audio)
        return self._processor(
            waveform,
            sampling_rate=audio.sample_rate,
            return_tensors="pt",
        )

    def _decode(self, generated_ids: Any, **kwargs: Any) -> str:
        """Decode model output IDs to a text string.

        Default uses ``processor.batch_decode``.  Subclasses may override
        for custom special-token handling.
        """
        decoded = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return decoded[0] if decoded else ""

    def _build_result(self, text: str, **kwargs: Any) -> ASRResult:
        """Build the final ``ASRResult`` from decoded text.

        Subclasses may override to inject word-level timestamps,
        speaker IDs, or confidence scores.
        """
        from modern_asr.core.types import ASRResult, Segment

        return ASRResult(
            text=text.strip(),
            segments=[Segment(text=text.strip())],
            language=kwargs.get("language", self.config.language),
            model_id=self.model_id,
        )

    # --- Language helpers -----------------------------------------------

    def _map_language(self, lang: str | None) -> str | None:
        """Map a short language code to the model-specific tag."""
        if lang is None:
            return None
        return self.LANGUAGE_MAP.get(lang, lang)

    # --- Internal utilities ---------------------------------------------

    @staticmethod
    def _import_cls(dotted_path: str) -> Any:
        """Dynamically import a class from a dotted module path.

        Examples:
            ``_import_cls("transformers.AutoProcessor")``
            ``_import_cls("my_package.modeling.MyModel")``
        """
        module_path, _, class_name = dotted_path.rpartition(".")
        if not module_path:
            raise ValueError(
                f"Invalid dotted class path: {dotted_path!r}. "
                f"Expected 'module.submodule.ClassName'."
            )
        module = importlib.import_module(module_path)
        try:
            return getattr(module, class_name)
        except AttributeError as exc:
            raise ImportError(
                f"Class '{class_name}' not found in module '{module_path}'."
            ) from exc
