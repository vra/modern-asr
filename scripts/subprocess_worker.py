#!/usr/bin/env python3
"""Generic subprocess worker for dependency-isolated ASR models.

Runs in an isolated virtual environment with its own dependency versions,
while the main ``modern-asr`` process uses a different set.  Any registered
model can be loaded inside this worker — it is **not** model-specific.

Communication protocol (newline-delimited JSON over stdin/stdout):

Init line (required before ready signal)::

    {"model_id": "qwen3-asr-0.6b", "device": "cpu"}

Inference request::

    {"cmd": "infer", "audio": "/path/to/audio.wav", "language": "zh"}

Response::

    {"status": "ok", "text": "transcription result", "language": "zh"}
    {"status": "error", "error": "..."}
"""

from __future__ import annotations

import json
import os
import sys
import warnings

# Ensure the main modern-asr source tree is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _main() -> None:
    # Import modern-asr components inside main so that any import errors
    # are reported back to the parent process.
    from modern_asr import create_model
    from modern_asr.core.config import BackendConfig, ModelConfig

    # Expect a single init message on the first line telling us which
    # model variant to load.
    init_line = sys.stdin.readline()
    if not init_line:
        print(json.dumps({"status": "error", "error": "no init line"}), flush=True)
        sys.exit(1)

    try:
        init_msg = json.loads(init_line)
    except json.JSONDecodeError as exc:
        print(json.dumps({"status": "error", "error": f"bad init JSON: {exc}"}), flush=True)
        sys.exit(1)

    model_id = init_msg.get("model_id")
    if not model_id:
        print(json.dumps({"status": "error", "error": "init missing model_id"}), flush=True)
        sys.exit(1)

    device = init_msg.get("device", "cpu")

    try:
        model = create_model(
            model_id,
            config=ModelConfig(model_id=model_id),
            backend=BackendConfig(device=device),
        )
        model.load()
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"load failed: {exc}"}), flush=True)
        sys.exit(1)

    # Signal readiness
    print(json.dumps({"status": "ready"}), flush=True)

    # Event loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"status": "error", "error": f"bad JSON: {exc}"}), flush=True)
            continue

        cmd = req.get("cmd")
        if cmd == "shutdown":
            print(json.dumps({"status": "ok"}), flush=True)
            break

        if cmd == "infer":
            audio_path = req.get("audio", "")
            language = req.get("language", "auto")
            try:
                from modern_asr.core.types import AudioInput
                audio_input = AudioInput(data=audio_path)
                result = model.transcribe(audio_input, language=language)
                print(
                    json.dumps(
                        {"status": "ok", "text": result.text, "language": result.language},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:
                import traceback
                print(json.dumps({"status": "error", "error": str(exc), "traceback": traceback.format_exc()}), flush=True)
        else:
            print(json.dumps({"status": "error", "error": f"unknown cmd: {cmd}"}), flush=True)


if __name__ == "__main__":
    _main()
