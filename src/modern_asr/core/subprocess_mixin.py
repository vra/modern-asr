"""Reusable mixin for subprocess-isolated ASR models.

Some upstream model packages pin dependency versions that conflict with the
main ``modern-asr`` environment (e.g. ``transformers==4.57.6`` vs
``transformers>=5.7.0``).  Rather than forcing users to choose one environment,
this mixin lets a model adapter transparently spawn a dedicated Python
subprocess with its own virtual environment and communicate with it via
newline-delimited JSON over stdin/stdout.

The worker script (``scripts/subprocess_worker.py``) is **fully generic** — it
uses ``create_model()`` and ``model.transcribe()``, so any registered model can
run inside it.  The only model-specific pieces are:

1. **Conflict detection** — a callable that returns ``True`` when the current
   environment is incompatible (checked at ``load()`` time).
2. **Venv path** — where the isolated environment lives.

Usage in a model adapter::

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
            if self._subprocess_backend is not None:
                return self._subprocess_transcribe(audio_path, **kwargs)
            return super().transcribe(audio, **kwargs)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from modern_asr.backends.subprocess_backend import SubprocessBackend
    from modern_asr.core.types import ASRResult


class SubprocessIsolatedMixin:
    """Mixin that adds subprocess-isolation capabilities to an ASR model.

    Subclasses declare *class attributes* (no boilerplate methods required).
    """

    # ------------------------------------------------------------------ #
    # Configuration — override in subclass
    # ------------------------------------------------------------------ #

    SUBPROCESS_VENV: str | None = None
    """Path to the isolated virtual environment.

    May be absolute or relative to the project root.
    """

    SUBPROCESS_ENV_VAR: str | None = None
    """Environment variable name that can override ``SUBPROCESS_VENV``.

    Example: ``"MODERN_ASR_QWEN_VENV"``.
    """

    SUBPROCESS_CHECK: Callable[[], bool] | None = None
    """Callable that returns ``True`` when the current environment has a
    conflicting dependency version and subprocess isolation is required."""

    SUBPROCESS_WORKER: str | None = None
    """Path to the worker script, relative to project root or absolute.

    Defaults to ``scripts/subprocess_worker.py``.
    """

    # ------------------------------------------------------------------ #
    # Internal state
    # ------------------------------------------------------------------ #

    _subprocess_backend: "SubprocessBackend" | None = None

    # ------------------------------------------------------------------ #
    # Loading helpers
    # ------------------------------------------------------------------ #

    def _try_native_then_subprocess(self, native_load: Callable[[], None]) -> None:
        """Try native loading; fall back to subprocess if conflict detected.

        Args:
            native_load: Callable that performs the native (in-process) model
                loading, e.g. ``super().load``.
        """
        check = getattr(self, "SUBPROCESS_CHECK", None)
        if check is not None and check():
            try:
                native_load()
                return
            except Exception:
                self._load_via_subprocess()
        else:
            native_load()

    def _load_via_subprocess(self) -> None:
        """Spawn a subprocess worker in an isolated virtual environment."""
        from modern_asr.backends.subprocess_backend import SubprocessBackend

        project_root = Path(__file__).resolve().parents[3]
        model_id = getattr(self, "model_id", "unknown")

        # --- discover venv ------------------------------------------------
        env_var = getattr(self, "SUBPROCESS_ENV_VAR", None)
        venv = getattr(self, "SUBPROCESS_VENV", None)

        venv_candidates: list[str] = []
        if env_var:
            venv_candidates.append(os.environ.get(env_var, ""))
        if venv:
            if os.path.isabs(venv):
                venv_candidates.append(venv)
            else:
                venv_candidates.append(str(project_root / venv))
        # fallback convention: .venv_{model_slug}
        slug = model_id.replace("-", "_")
        venv_candidates.append(str(project_root / f".venv_{slug}"))

        python_exe: str | None = None
        for candidate in venv_candidates:
            if candidate and os.path.isfile(os.path.join(candidate, "bin", "python")):
                python_exe = os.path.join(candidate, "bin", "python")
                break

        if python_exe is None:
            searched = [c for c in venv_candidates if c]
            raise RuntimeError(
                f"{model_id} requires an isolated virtual environment because "
                f"the current environment has conflicting dependencies.\n\n"
                f"Searched: {searched}\n\n"
                f"Please create one:\n"
                f"  cd {project_root}\n"
                f"  python3.10 -m venv {venv or '.venv_' + slug}\n"
                f"  source {venv or '.venv_' + slug}/bin/activate\n"
                f"  pip install <model-specific-deps>\n\n"
                f"Or set {env_var}=<path>"
                if env_var
                else ""
            )

        # --- discover worker script ---------------------------------------
        worker = getattr(self, "SUBPROCESS_WORKER", None)
        if worker:
            worker_script = worker if os.path.isabs(worker) else str(project_root / worker)
        else:
            worker_script = str(project_root / "scripts" / "subprocess_worker.py")

        if not os.path.isfile(worker_script):
            raise RuntimeError(f"Worker script not found: {worker_script}")

        # --- resolve device for init payload ------------------------------
        resolve_device = getattr(self, "_resolve_device", None)
        device = resolve_device() if resolve_device else "cpu"

        # --- spawn backend ------------------------------------------------
        self._subprocess_backend = SubprocessBackend(
            python_executable=python_exe,
            worker_script=worker_script,
            env={
                "MODERN_ASR_CACHE_DIR": os.environ.get(
                    "MODERN_ASR_CACHE_DIR",
                    str(Path.home() / ".cache" / "modern-asr"),
                ),
            },
            init_payload={"model_id": model_id, "device": device},
        )
        self._is_loaded = True

    # ------------------------------------------------------------------ #
    # Inference helpers
    # ------------------------------------------------------------------ #

    def _subprocess_transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        **kwargs: Any,
    ) -> "ASRResult":
        """Delegate transcription to the subprocess worker.

        Args:
            audio_path: Absolute path to the audio file.
            language: Target language (or ``None`` for auto).
            **kwargs: Unused — kept for API compatibility.

        Returns:
            An ``ASRResult`` built via ``self._build_result``.
        """
        backend = getattr(self, "_subprocess_backend", None)
        if backend is None:
            raise RuntimeError(
                "Subprocess backend not initialized; call load() first."
            )

        resp = backend.infer(audio=audio_path, language=language or "auto")
        if resp.get("status") != "ok":
            raise RuntimeError(f"Worker error: {resp.get('error', 'unknown')}")

        text = resp.get("text", "")
        detected_lang = resp.get("language", language or "auto")
        kw = dict(kwargs)
        kw.pop("language", None)
        # _build_result is provided by AudioLLMModel / ASRModel
        build_result = getattr(self, "_build_result", None)
        if build_result is None:
            from modern_asr.core.types import ASRResult, Segment
            return ASRResult(
                text=text.strip(),
                segments=[Segment(text=text.strip())],
                language=detected_lang,
                model_id=getattr(self, "model_id", "unknown"),
            )
        return build_result(text, language=detected_lang, **kw)

    def shutdown_subprocess(self) -> None:
        """Gracefully terminate the subprocess worker if running."""
        backend = getattr(self, "_subprocess_backend", None)
        if backend is not None:
            backend.shutdown()
            self._subprocess_backend = None
