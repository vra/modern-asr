"""Command-line interface for Modern ASR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from modern_asr.core.pipeline import ASRPipeline
from modern_asr.core.registry import list_models


def _print_models(args: argparse.Namespace) -> int:
    models = list_models(
        language=args.language,
        mode=args.mode,
    )
    if args.json:
        print(json.dumps(models, indent=2, ensure_ascii=False))
    else:
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


def _transcribe(args: argparse.Namespace) -> int:
    pipe = ASRPipeline(
        model_id=args.model,
    )
    result = pipe(
        audio=args.audio,
        task=args.task,
        language=args.language,
    )

    if args.output:
        out_path = Path(args.output)
        if args.output.endswith(".json"):
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
        else:
            out_path.write_text(result.text, encoding="utf-8")
        print(f"Saved to {out_path}")
    else:
        print(result.text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="modern-asr",
        description="Unified CLI for LLM-based Automatic Speech Recognition.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # list
    list_parser = subparsers.add_parser("list", help="List registered models")
    list_parser.add_argument("--language", default=None, help="Filter by language code")
    list_parser.add_argument("--mode", default=None, help="Filter by mode (transcribe, translate, diarize, emotion, event)")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    list_parser.set_defaults(func=_print_models)

    # transcribe
    trans_parser = subparsers.add_parser("transcribe", help="Transcribe an audio file")
    trans_parser.add_argument("audio", help="Path to audio file")
    trans_parser.add_argument("-m", "--model", required=True, help="Model ID (e.g. sensevoice-small)")
    trans_parser.add_argument("-t", "--task", default="transcribe", help="Task: transcribe, translate, diarize, emotion, event")
    trans_parser.add_argument("-l", "--language", default="auto", help="Language code")
    trans_parser.add_argument("-o", "--output", default=None, help="Output file path (default: stdout)")
    trans_parser.set_defaults(func=_transcribe)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
