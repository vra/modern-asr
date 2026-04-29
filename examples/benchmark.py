"""Benchmark multiple ASR models on the same audio file."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from modern_asr import ASRPipeline, list_models


def benchmark(
    audio_path: str,
    models: list[str] | None = None,
    language: str = "auto",
) -> dict[str, dict]:
    """Run transcription on multiple models and collect metrics."""
    if models is None:
        models = [m["model_id"] for m in list_models()]

    results: dict[str, dict] = {}
    pipe = ASRPipeline()

    for model_id in models:
        print(f"Benchmarking: {model_id} ...")
        try:
            pipe.switch_model(model_id)
            t0 = time.perf_counter()
            result = pipe(audio_path, language=language)
            elapsed = time.perf_counter() - t0
            results[model_id] = {
                "text": result.text,
                "time": elapsed,
                "success": True,
                "language": result.language,
            }
        except Exception as exc:
            results[model_id] = {
                "text": "",
                "time": 0.0,
                "success": False,
                "error": str(exc),
            }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ASR models")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--models", nargs="+", default=None, help="Model IDs to benchmark")
    parser.add_argument("--language", default="auto", help="Language code")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    results = benchmark(args.audio, models=args.models, language=args.language)

    import json
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
