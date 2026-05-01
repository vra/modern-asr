"""Auto-installation of missing dependencies, git repos, and model weights.

Users should only need to type the model name — everything else is handled
automatically on first use.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

from modern_asr.utils.log import get_logger

logger = get_logger(__name__)

_CACHE_ROOT = Path.home() / ".cache" / "modern-asr"


def _run(cmd: list[str]) -> None:
    """Run a subprocess command, suppressing output."""
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ensure_pypi(pkg_spec: str, import_name: str | None = None) -> None:
    """Auto-install a PyPI (or git+https) package if the import fails.

    Args:
        pkg_spec: pip install spec, e.g. ``"transformers>=4.40.0"`` or
            ``"git+https://github.com/org/repo.git"``.
        import_name: Python module name to test.  If ``None``, derived from
            ``pkg_spec`` by taking the first segment before ``[``, ``>``, ``=``.
    """
    if import_name is None:
        # "transformers>=4.40"  -> "transformers"
        # "nemo-toolkit[asr]"   -> "nemo-toolkit"
        # "git+https://..."     -> last path segment
        raw = pkg_spec.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip()
        if raw.startswith("git+"):
            raw = raw.rstrip("/").split("/")[-1].replace(".git", "")
        import_name = raw.replace("-", "_")
    try:
        importlib.import_module(import_name)
    except ImportError:
        logger.info("Auto-installing %s ...", pkg_spec)
        _run([sys.executable, "-m", "pip", "install", pkg_spec])


def ensure_git(url: str, name: str | None = None) -> Path:
    """Clone a git repo to the cache directory if not already present.

    Returns the repo root path.
    """
    if name is None:
        name = url.rstrip("/").split("/")[-1].replace(".git", "")
    dest = _CACHE_ROOT / "repos" / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Auto-cloning %s ...", url)
        _run(["git", "clone", "--depth", "1", url, str(dest)])
    return dest


def ensure_hf(repo_id: str, name: str | None = None) -> Path:
    """Download HuggingFace model weights to the cache directory if missing.

    Returns the local directory path.
    """
    ensure_pypi("huggingface-hub>=0.20.0", "huggingface_hub")
    from huggingface_hub import snapshot_download

    if name is None:
        name = repo_id.replace("/", "--")
    dest = _CACHE_ROOT / "models" / name
    if not dest.exists() or not any(dest.iterdir()):
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Auto-downloading %s ...", repo_id)
        token = os.environ.get("HF_TOKEN")
        kwargs: dict[str, object] = {"local_dir": str(dest)}
        if token:
            kwargs["token"] = token
        snapshot_download(repo_id, **kwargs)
    return dest
