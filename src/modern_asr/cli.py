"""Command-line interface for Modern ASR.

Usage::

    masr list                          # List all models
    masr run <model> <audio>           # Transcribe audio
    masr batch <model> <dir>           # Batch process directory
    masr bench <audio>                 # Benchmark models
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from modern_asr.core.config import BackendConfig
from modern_asr.core.pipeline import ASRPipeline
from modern_asr.core.registry import list_models


def _print_models(args: argparse.Namespace) -> int:
    """List registered models."""
    models = list_models(
        language=args.language,
        mode=args.mode,
    )
    if args.json:
        print(json.dumps(models, indent=2, ensure_ascii=False))
        return 0

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Registered ASR Models")
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Languages")
    table.add_column("Modes")
    table.add_column("Module")

    for m in models:
        table.add_row(
            m["model_id"],
            ", ".join(sorted(m["supported_languages"]))[:40],
            ", ".join(sorted(m["supported_modes"])),
            m["module"],
        )
    console.print(table)
    return 0


def _run(args: argparse.Namespace) -> int:
    """Transcribe a single audio file."""
    from rich.console import Console

    console = Console()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        console.print(f"[red]Error:[/red] File not found: {audio_path}")
        return 1

    backend = BackendConfig(
        device=args.device,
        dtype=args.dtype,
    )

    console.print(f"Loading [bold cyan]{args.model}[/bold cyan]...")
    pipe = ASRPipeline(model_id=args.model, backend=backend)

    console.print(f"Transcribing [bold]{audio_path.name}[/bold]...")
    t0 = time.time()
    result = pipe(
        audio=str(audio_path),
        task=args.task,
        language=args.language,
    )
    elapsed = time.time() - t0

    console.print(f"\n[green]Done in {elapsed:.2f}s[/green]")

    if args.output:
        _write_output(result, args.output, args.format)
    else:
        print(result.text)

    return 0


def _batch(args: argparse.Namespace) -> int:
    """Batch transcribe all audio files in a directory."""
    from rich.console import Console
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

    console = Console()
    src_dir = Path(args.source)
    if not src_dir.is_dir():
        console.print(f"[red]Error:[/red] Not a directory: {src_dir}")
        return 1

    exts = {e.strip().lower() for e in args.ext.split(",")}
    files = sorted(
        f for f in src_dir.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    )
    if not files:
        console.print(f"[yellow]Warning:[/yellow] No audio files found in {src_dir}")
        return 0

    backend = BackendConfig(device=args.device, dtype=args.dtype)
    pipe = ASRPipeline(model_id=args.model, backend=backend)

    out_dir = Path(args.output) if args.output else src_dir / "transcripts"
    out_dir.mkdir(exist_ok=True)

    console.print(f"Processing {len(files)} files with [bold cyan]{args.model}[/bold cyan]...")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Transcribing...", total=len(files))
        for f in files:
            try:
                result = pipe(str(f), task=args.task, language=args.language)
                out_path = out_dir / f"{f.stem}.txt"
                out_path.write_text(result.text, encoding="utf-8")
            except Exception as exc:
                console.print(f"[red]Failed on {f.name}:[/red] {exc}")
            progress.advance(task)

    console.print(f"\n[green]Saved to {out_dir}[/green]")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    """Benchmark multiple models on the same audio file."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        console.print(f"[red]Error:[/red] File not found: {audio_path}")
        return 1

    models = [m.strip() for m in args.models.split(",")]
    backend = BackendConfig(device=args.device, dtype=args.dtype)

    console.print(f"Benchmarking {len(models)} models on [bold]{audio_path.name}[/bold]\n")

    results: list[dict[str, Any]] = []
    for model_id in models:
        console.print(f"Running [cyan]{model_id}[/cyan]...", end=" ")
        try:
            pipe = ASRPipeline(model_id=model_id, backend=backend)
            t0 = time.time()
            result = pipe(str(audio_path), language=args.language)
            elapsed = time.time() - t0
            console.print(f"[green]{elapsed:.2f}s[/green]")
            results.append({
                "model": model_id,
                "time": elapsed,
                "text": result.text[:200],
            })
            pipe.unload()
        except Exception as exc:
            console.print(f"[red]FAILED: {exc}[/red]")
            results.append({"model": model_id, "time": None, "text": str(exc)})

    table = Table(title="Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("Time", justify="right")
    table.add_column("Text (first 200 chars)")
    for r in results:
        time_str = f"{r['time']:.2f}s" if r["time"] is not None else "N/A"
        table.add_row(r["model"], time_str, r["text"])
    console.print()
    console.print(table)
    return 0


def _write_output(result: Any, output: str, fmt: str | None) -> None:
    """Write result to file in the requested format."""
    out_path = Path(output)
    fmt = fmt or out_path.suffix.lstrip(".") or "txt"

    if fmt == "json":
        out_path.write_text(
            json.dumps(
                {
                    "text": result.text,
                    "language": result.language,
                    "model_id": result.model_id,
                    "segments": [
                        {
                            "text": s.text,
                            "start": s.start,
                            "end": s.end,
                            "words": [
                                {"text": w.text, "start": w.start, "end": w.end}
                                for w in s.words
                            ],
                        }
                        for s in result.segments
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    elif fmt in ("srt", "vtt"):
        out_path.write_text(_to_subtitle(result, fmt), encoding="utf-8")
    else:
        out_path.write_text(result.text, encoding="utf-8")

    print(f"Saved to {out_path}")


def _to_subtitle(result: Any, fmt: str) -> str:
    """Convert result to SRT or VTT subtitle format."""
    lines: list[str] = []
    if fmt == "vtt":
        lines.append("WEBVTT\n")
    for i, seg in enumerate(result.segments, start=1):
        start = _format_time(seg.start or 0.0, fmt)
        end = _format_time(seg.end or 0.0, fmt)
        if fmt == "srt":
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(seg.text)
            lines.append("")
        else:
            lines.append(f"{start} --> {end}")
            lines.append(seg.text)
            lines.append("")
    return "\n".join(lines)


def _format_time(seconds: float, fmt: str) -> str:
    """Format seconds as SRT (HH:MM:SS,mmm) or VTT (HH:MM:SS.mmm) time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    sep = "," if fmt == "srt" else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="masr",
        description="Unified CLI for LLM-based Automatic Speech Recognition.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------ #
    # list
    # ------------------------------------------------------------------ #
    list_parser = subparsers.add_parser("list", help="List registered models")
    list_parser.add_argument("--language", default=None, help="Filter by language code")
    list_parser.add_argument("--mode", default=None, help="Filter by mode (transcribe, translate, diarize, emotion, event)")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    list_parser.set_defaults(func=_print_models)

    # ------------------------------------------------------------------ #
    # run
    # ------------------------------------------------------------------ #
    run_parser = subparsers.add_parser("run", help="Transcribe an audio file")
    run_parser.add_argument("model", help="Model ID (e.g. sensevoice-small)")
    run_parser.add_argument("audio", help="Path to audio file")
    run_parser.add_argument("-t", "--task", default="transcribe", help="Task: transcribe, translate, diarize, emotion, event")
    run_parser.add_argument("-l", "--language", default="auto", help="Language code")
    run_parser.add_argument("-o", "--output", default=None, help="Output file path")
    run_parser.add_argument("-f", "--format", default=None, choices=["txt", "json", "srt", "vtt"], help="Output format (inferred from --output suffix if omitted)")
    run_parser.add_argument("--device", default="auto", help="Device: auto, cuda, mps, cpu")
    run_parser.add_argument("--dtype", default="auto", help="Dtype: auto, float16, bfloat16, float32")
    run_parser.set_defaults(func=_run)

    # ------------------------------------------------------------------ #
    # batch
    # ------------------------------------------------------------------ #
    batch_parser = subparsers.add_parser("batch", help="Batch transcribe a directory")
    batch_parser.add_argument("model", help="Model ID")
    batch_parser.add_argument("source", help="Source directory containing audio files")
    batch_parser.add_argument("-o", "--output", default=None, help="Output directory (default: <source>/transcripts/)")
    batch_parser.add_argument("--ext", default=".wav,.mp3,.flac,.m4a,.ogg", help="Comma-separated file extensions to process")
    batch_parser.add_argument("-t", "--task", default="transcribe", help="Task")
    batch_parser.add_argument("-l", "--language", default="auto", help="Language code")
    batch_parser.add_argument("--device", default="auto", help="Device")
    batch_parser.add_argument("--dtype", default="auto", help="Dtype")
    batch_parser.set_defaults(func=_batch)

    # ------------------------------------------------------------------ #
    # bench
    # ------------------------------------------------------------------ #
    bench_parser = subparsers.add_parser("bench", help="Benchmark multiple models")
    bench_parser.add_argument("audio", help="Audio file to benchmark on")
    bench_parser.add_argument("--models", default="sensevoice-small,qwen3-asr-0.6b,whisper-small", help="Comma-separated model IDs")
    bench_parser.add_argument("-l", "--language", default="auto", help="Language code")
    bench_parser.add_argument("--device", default="auto", help="Device")
    bench_parser.add_argument("--dtype", default="auto", help="Dtype")
    bench_parser.set_defaults(func=_benchmark)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
