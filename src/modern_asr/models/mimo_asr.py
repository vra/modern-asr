"""MiMo-V2.5-ASR adapter (Xiaomi / 小米).

Models:
    - mimo-asr-v2.5: End-to-end ASR, supports dialects, code-switching, noise,
      multi-speaker, song lyrics, and knowledge-intensive content.

Architecture
------------
MiMo-V2.5-ASR is an **Audio-LLM** model with two components:

1. **MiMo-Audio-Tokenizer** — encodes raw audio into discrete tokens.
2. **MiMo-V2.5-ASR** (8B causal LM) — decodes audio tokens + optional
   language tags into text.

Because the inference code lives in the official GitHub repo (not on
PyPI or inside ``transformers``), this adapter locates a local clone,
injects ``src/`` into ``sys.path``, and wraps the ``MimoAudio`` class.

Prerequisites
-------------
* Python >= 3.12 (upstream tests on 3.12)
* CUDA >= 12.0
* ``flash-attn >= 2.7.4`` (strongly recommended for speed)
* Local clone of the official repo + downloaded weights

Quick-start::

    git clone https://github.com/XiaomiMiMo/MiMo-V2.5-ASR.git
    cd MiMo-V2.5-ASR
    pip install -r requirements.txt

    huggingface-cli download XiaomiMiMo/MiMo-V2.5-ASR \
        --local-dir models/MiMo-V2.5-ASR
    huggingface-cli download XiaomiMiMo/MiMo-Audio-Tokenizer \
        --local-dir models/MiMo-Audio-Tokenizer

    # In Python
    from modern_asr import ASRPipeline
    pipe = ASRPipeline("mimo-asr-v2.5")
    result = pipe("audio.wav", language="zh")

License: Apache-2.0
References:
    - https://github.com/XiaomiMiMo/MiMo-V2.5-ASR
    - https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR
    - https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer
"""

from __future__ import annotations

import os
import sys
from typing import Any

from modern_asr.core.audio_llm import AudioLLMModel
from modern_asr.core.config import BackendConfig, ModelConfig
from modern_asr.core.registry import register_model
from modern_asr.core.types import ASRResult


def _check_deps() -> None:
    from modern_asr.utils.auto_install import ensure_pypi

    ensure_pypi("torch>=2.0")
    ensure_pypi("transformers>=4.40.0")


def _patch_mimo_flash_attn(repo_root: str) -> None:
    """Make flash_attn optional in the MiMo repo by injecting a PyTorch fallback."""
    import pathlib

    target = pathlib.Path(repo_root) / "src" / "mimo_audio_tokenizer" / "modeling_audio_tokenizer.py"
    if not target.exists():
        return

    src = target.read_text(encoding="utf-8")
    if "flash_attn_optional" in src:
        return  # Already patched

    # Replace the unconditional import with a try/except + fallback
    old_import = "from flash_attn import flash_attn_varlen_func"
    new_import = '''try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    import torch.nn.functional as _F
    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False, window_size=(-1, -1), **kwargs):
        """PyTorch fallback for flash_attn_varlen_func."""
        batch_size = len(cu_seqlens_q) - 1
        outputs = []
        for i in range(batch_size):
            qi = q[cu_seqlens_q[i]:cu_seqlens_q[i+1]].unsqueeze(0).transpose(1, 2)
            ki = k[cu_seqlens_k[i]:cu_seqlens_k[i+1]].unsqueeze(0).transpose(1, 2)
            vi = v[cu_seqlens_k[i]:cu_seqlens_k[i+1]].unsqueeze(0).transpose(1, 2)
            out = _F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal)
            out = out.transpose(1, 2).squeeze(0)
            outputs.append(out)
        return torch.cat(outputs, dim=0)
    # mark as patched
    flash_attn_varlen_func._flash_attn_optional = True  # type: ignore[attr-defined]
'''

    if old_import in src:
        src = src.replace(old_import, new_import)
        target.write_text(src, encoding="utf-8")


def _patch_mimo_rotary_embedding(repo_root: str) -> None:
    """Patch MiMo's RotaryEmbedding for transformers 5.7.0 compatibility.

    transformers 5.7.0's ``_init_weights`` expects ``compute_default_rope_parameters``
    on every ``RotaryEmbedding`` with ``rope_type == 'default'``.  MiMo's custom
    ``RotaryEmbedding`` lacks this method and a ``config`` attribute, so
    ``from_pretrained`` crashes after weight loading.

    We inject the missing method and a minimal ``config`` property into the
    class by rewriting the source file.
    """
    import pathlib

    target = pathlib.Path(repo_root) / "src" / "mimo_audio_tokenizer" / "modeling_audio_tokenizer.py"
    if not target.exists():
        return

    src = target.read_text(encoding="utf-8")
    if "# PATCH: transformers 5.7.0 rotary embedding" in src:
        return  # Already patched

    # 1. Store base/dim in __init__ so compute_default_rope_parameters can use them
    old_init = """        self.max_seq_len = max_seq_len
        self.rope_type = rope_type

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            device=device, base=base, dim=dim
        )"""

    new_init = """        self.max_seq_len = max_seq_len
        self.rope_type = rope_type
        self.base = base
        self.dim = dim

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            device=device, base=base, dim=dim
        )"""

    if old_init in src:
        src = src.replace(old_init, new_init)

    # 2. Insert compute_default_rope_parameters and config property before forward()
    old_forward = "    @torch.no_grad()\n    @dynamic_rope_update\n    def forward(self, x, position_ids):"

    new_methods = '''    # PATCH: transformers 5.7.0 rotary embedding
    @property
    def config(self):
        """Dummy config for transformers 5.7.0 _init_weights compatibility."""
        class _DummyConfig:
            rope_parameters = {"rope_theta": getattr(self, "base", 10000.0)}
        return _DummyConfig()

    def compute_default_rope_parameters(self, config=None, device=None, seq_len=None):
        """Return already-computed inv_freq (transformers 5.7.0 expects this)."""
        return self.inv_freq, self.attention_scaling

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):'''

    if old_forward in src:
        src = src.replace(old_forward, new_methods)

    target.write_text(src, encoding="utf-8")


@register_model("mimo-asr-v2.5")
class MiMoASRV25(AudioLLMModel):
    """MiMo-V2.5-ASR: Xiaomi's open-source ASR for complex real-world scenarios."""

    MODEL_CARD = "https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR"
    SUPPORTED_LANGUAGES = {"zh", "en", "yue", "wuu", "nan", "cmn", "auto", "multi"}
    SUPPORTED_MODES = {"transcribe", "diarize"}
    REQUIREMENTS = ["torch", "transformers"]

    # Language tags used by MiMo's asr_sft() method
    _LANGUAGE_TAGS = {
        "zh": "<chinese>",
        "en": "<english>",
    }

    _REPO_URL = "https://github.com/XiaomiMiMo/MiMo-V2.5-ASR.git"

    def __init__(
        self,
        config: ModelConfig,
        backend: BackendConfig | None = None,
    ) -> None:
        # We do NOT call AudioLLMModel.__init__ path resolution because MiMo
        # uses a local directory layout rather than a pure HF hub ID.
        super().__init__(config, backend)
        self._model_dir = self._resolve_model_dir()
        self._tokenizer_dir = self._resolve_tokenizer_dir()

    # ------------------------------------------------------------------ #
    # Path resolution
    # ------------------------------------------------------------------ #

    def _resolve_model_dir(self) -> str:
        if self.config.model_path:
            return str(self.config.model_path)
        candidates = [
            "./models/MiMo-V2.5-ASR",
            "./MiMo-V2.5-ASR/models/MiMo-V2.5-ASR",
            "/tmp/MiMo-V2.5-ASR/models/MiMo-V2.5-ASR",
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        return candidates[0]

    def _resolve_tokenizer_dir(self) -> str:
        model = self._model_dir
        candidates = [
            os.path.join(os.path.dirname(model), "MiMo-Audio-Tokenizer"),
            "./models/MiMo-Audio-Tokenizer",
            "/tmp/MiMo-V2.5-ASR/models/MiMo-Audio-Tokenizer",
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        return candidates[0]

    def _find_repo_root(self) -> str | None:
        """Walk upward from model_dir to find the repo root containing src/."""
        path = os.path.abspath(self._model_dir)
        for _ in range(5):
            if os.path.isdir(os.path.join(path, "src", "mimo_audio")):
                return path
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
        return None

    @property
    def model_id(self) -> str:
        return "mimo-asr-v2.5"

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        from modern_asr.utils.log import get_logger

        logger = get_logger(__name__)
        logger.info("Loading %s", self.model_id)

        _check_deps()

        from modern_asr.utils.auto_install import ensure_git, ensure_hf

        # 1. Auto-clone the official repo if missing
        repo_root = ensure_git(self._REPO_URL, "MiMo-V2.5-ASR")
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        # 1b. Patch flash_attn dependency to optional fallback
        _patch_mimo_flash_attn(str(repo_root))

        # 1c. Patch RotaryEmbedding for transformers 5.7.0 compatibility
        _patch_mimo_rotary_embedding(str(repo_root))

        # 1d. Patch GenerationMixin for transformers 5.7.0 compatibility
        # In transformers >=5.x, PreTrainedModel no longer inherits from
        # GenerationMixin. MiMoAudioForCausalLM must explicitly mix it in.
        import transformers.generation.utils as _gen_utils
        from src.mimo_audio.modeling_mimo_audio import MiMoAudioForCausalLM
        if not issubclass(MiMoAudioForCausalLM, _gen_utils.GenerationMixin):
            MiMoAudioForCausalLM.__bases__ = MiMoAudioForCausalLM.__bases__ + (_gen_utils.GenerationMixin,)

        # 1e. Patch GenerationMixin._has_unfinished_sequences for backward compat
        # MiMo's custom generate() passes cur_len/max_length kwargs that were
        # removed in transformers 5.7.0.
        _orig_has_unfinished = _gen_utils.GenerationMixin._has_unfinished_sequences
        def _patched_has_unfinished(self, this_peer_finished, synced_gpus, device, **kwargs):
            return _orig_has_unfinished(self, this_peer_finished, synced_gpus, device)
        _gen_utils.GenerationMixin._has_unfinished_sequences = _patched_has_unfinished

        # 1f. Add missing attributes expected by MiMo's prepare_inputs_for_generation
        MiMoAudioForCausalLM._supports_cache_class = False  # type: ignore[attr-defined]

        # 2. Auto-download HF weights if missing
        model_dir = ensure_hf("XiaomiMiMo/MiMo-V2.5-ASR", "mimo-v2.5-asr")
        tokenizer_dir = ensure_hf(
            "XiaomiMiMo/MiMo-Audio-Tokenizer", "mimo-audio-tokenizer"
        )

        from src.mimo_audio.mimo_audio import MimoAudio

        device = self._resolve_device()
        self._model = MimoAudio(str(model_dir), str(tokenizer_dir), device=device)
        self._is_loaded = True
        logger.info("%s ready on %s", self.model_id, device)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def transcribe(self, audio: Any, **kwargs: Any) -> ASRResult:
        self._ensure_loaded()
        audio_path = self._audio_to_file(audio)

        lang = kwargs.get("language", self.config.language)
        audio_tag = self._LANGUAGE_TAGS.get(lang or "", "")

        text = self._model.asr_sft(audio_path, audio_tag=audio_tag)
        return self._build_result(text, **kwargs)

    def diarize(self, audio: Any, **kwargs: Any) -> ASRResult:
        """MiMo-V2.5-ASR supports multi-speaker transcription via ASR."""
        return self.transcribe(audio, **kwargs)
