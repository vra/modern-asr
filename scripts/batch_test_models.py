#!/usr/bin/env python3
"""Batch test all registered ASR models against a single audio file.

Usage::

    cd /home/ws/ws/projects/modern-asr
    PYTHONPATH=src python scripts/batch_test_models.py ~/1111.wav

Results are written to ``batch_test_results.md`` in the current directory.

Each model runs in a fresh subprocess so that:
* Memory is freed automatically between models.
* A hung model can be killed after a timeout without crashing the whole run.
* Dependency-conflicted models (e.g. Qwen3-ASR) can use their isolated venv
  without interference.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Ensure src/ is on path before any modern_asr imports in the main process
# --------------------------------------------------------------------------- #
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _model_worker(model_id: str, audio_path: str, conn: multiprocessing.connection.Connection) -> None:
    """Run a single model in an isolated process."""
    # Re-establish PYTHONPATH in spawned process
    sys.path.insert(0, str(_SRC))

    # Mirror the user's cache / endpoint settings
    os.environ.setdefault("MODERN_ASR_CACHE_DIR", "/mnt/hdd/.cache/modern-asr")
    os.environ.setdefault("HF_HOME", "/mnt/hdd/.cache/huggingface")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # Force CPU-only so that large models do not OOM a shared GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    result: dict[str, Any] = {
        "model_id": model_id,
        "status": "pending",
        "text": "",
        "language": "",
        "load_seconds": 0.0,
        "infer_seconds": 0.0,
        "total_seconds": 0.0,
        "error": "",
    }

    t0 = time.time()
    try:
        from modern_asr import create_model
        from modern_asr.core.config import BackendConfig, ModelConfig
        from modern_asr.core.types import AudioInput

        cfg = ModelConfig(model_id=model_id)
        backend = BackendConfig(device="cpu")
        model = create_model(model_id, config=cfg, backend=backend)

        t_load_start = time.time()
        model.load()
        result["load_seconds"] = time.time() - t_load_start

        audio = AudioInput(data=audio_path)
        t_infer_start = time.time()
        out = model.transcribe(audio, language="auto")
        result["infer_seconds"] = time.time() - t_infer_start

        result["text"] = out.text
        result["language"] = out.language or ""
        result["status"] = "success"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    result["total_seconds"] = time.time() - t0
    conn.send(result)
    conn.close()


def test_model(model_id: str, audio_path: str, timeout: float) -> dict[str, Any]:
    """Test one model with a hard timeout, using a spawned process."""
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_model_worker, args=(model_id, audio_path, child_conn))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join(timeout=5.0)
        if p.is_alive():
            p.kill()
            p.join(timeout=2.0)
        return {
            "model_id": model_id,
            "status": "timeout",
            "text": "",
            "language": "",
            "load_seconds": timeout,
            "infer_seconds": 0.0,
            "total_seconds": timeout,
            "error": f"Killed after {timeout}s timeout",
        }

    try:
        return parent_conn.recv()
    except EOFError:
        return {
            "model_id": model_id,
            "status": "crashed",
            "text": "",
            "language": "",
            "load_seconds": 0.0,
            "infer_seconds": 0.0,
            "total_seconds": timeout,
            "error": "Worker process crashed or produced no output",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-test all ASR models")
    parser.add_argument("audio", nargs="?", default="~/1111.wav", help="Input audio file")
    parser.add_argument("-o", "--output", default="batch_test_results.md", help="Output markdown file")
    parser.add_argument("-t", "--timeout", type=float, default=600.0, help="Per-model timeout in seconds")
    parser.add_argument("--models", default="", help="Comma-separated model IDs (default: all)")
    args = parser.parse_args()

    audio_path = os.path.expanduser(args.audio)
    if not os.path.isfile(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    from modern_asr import list_models

    all_models = list_models(mode="transcribe")
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = [m for m in all_models if m["model_id"] in wanted]
    else:
        models = all_models

    print(f"Testing {len(models)} model(s) against {audio_path}")
    print(f"Timeout per model: {args.timeout}s")
    print("-" * 60)

    results: list[dict[str, Any]] = []
    for info in models:
        model_id = info["model_id"]
        print(f"[{len(results)+1}/{len(models)}] {model_id} …", end=" ", flush=True)
        r = test_model(model_id, audio_path, args.timeout)
        results.append(r)
        icon = "✅" if r["status"] == "success" else "❌"
        snippet = (r["text"] or "")[:60].replace("\n", " ")
        print(f"{icon} {r['total_seconds']:.1f}s | {snippet}")

    # ------------------------------------------------------------------ #
    # Write Markdown report
    # ------------------------------------------------------------------ #
    report_path = Path(args.output)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Batch ASR Model Test Results\n\n")
        f.write(f"**Audio:** `{audio_path}`  \n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Timeout:** {args.timeout}s per model  \n\n")

        success = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - success
        f.write(f"**Summary:** {success} passed / {failed} failed / {len(results)} total  \n\n")

        f.write("## Quick Overview\n\n")
        f.write("| # | Model | Status | Load (s) | Infer (s) | Total (s) | Result |\n")
        f.write("|---|-------|--------|----------|-----------|-----------|--------|\n")
        for i, r in enumerate(results, 1):
            status = "✅" if r["status"] == "success" else "❌"
            text_short = (r["text"] or "").replace("\n", " ").replace("|", "\\|")[:60]
            f.write(
                f"| {i} | {r['model_id']} | {status} {r['status']} | "
                f"{r['load_seconds']:.1f} | {r['infer_seconds']:.1f} | "
                f"{r['total_seconds']:.1f} | {text_short} |\n"
            )

        f.write("\n## Details\n\n")
        for r in results:
            f.write(f"### {r['model_id']}\n\n")
            f.write(f"- **Status:** {r['status']}\n")
            f.write(f"- **Load time:** {r['load_seconds']:.2f}s\n")
            f.write(f"- **Inference time:** {r['infer_seconds']:.2f}s\n")
            f.write(f"- **Total time:** {r['total_seconds']:.2f}s\n")
            if r["status"] == "success":
                f.write(f"- **Detected language:** {r['language']}\n")
                f.write(f"- **Text:**\n")
                f.write(textwrap.indent(r["text"], "> "))
                f.write("\n")
            else:
                f.write("- **Error:**\n")
                f.write("```\n")
                f.write(r["error"])
                f.write("```\n")
            f.write("\n---\n\n")

    print("-" * 60)
    print(f"Report written to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
