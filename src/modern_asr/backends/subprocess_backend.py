"""Subprocess-based inference backend for dependency-isolated models.

Some upstream models pin conflicting dependency versions (e.g.
``transformers==4.57.6`` vs ``transformers>=5.7.0``).  Rather than forcing
users to choose one environment, this backend spawns a dedicated Python
subprocess with its own virtual environment and communicates with it via
newline-delimited JSON over stdin/stdout.

Usage::

    backend = SubprocessBackend(
        python_executable="/path/to/.venv_qwen310/bin/python",
        worker_script="/path/to/worker.py",
    )
    result = backend.infer(audio_path="/tmp/audio.wav", language="zh")
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


class SubprocessBackend:
    """Long-running subprocess backend with JSON-RPC-style communication."""

    def __init__(
        self,
        python_executable: str | Path,
        worker_script: str | Path,
        env: dict[str, str] | None = None,
        startup_timeout: float = 300.0,
        init_payload: dict[str, Any] | None = None,
    ) -> None:
        self._python = str(python_executable)
        self._worker = str(worker_script)
        self._env = {**os.environ, **(env or {})}
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._start(startup_timeout, init_payload)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _start(self, timeout: float, init_payload: dict[str, Any] | None) -> None:
        """Spawn the worker subprocess and wait for the ready signal."""
        self._proc = subprocess.Popen(
            [self._python, self._worker],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._env,
            bufsize=1,  # line-buffered
        )
        # Some workers need an init line (model_id, device, etc.) before
        # they can load and signal readiness.  Send it immediately.
        if init_payload is not None:
            self._send(init_payload)

        # Wait for the worker to print its ready line
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            assert self._proc.stdout is not None
            line = self._proc.stdout.readline()
            if not line:
                # Process died early – capture stderr for diagnostics
                stderr = self._proc.stderr.read() if self._proc.stderr else ""
                raise RuntimeError(
                    f"Subprocess worker died during startup.\n"
                    f"stderr: {stderr[:2000]}"
                )
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("status") == "ready":
                return
            if msg.get("status") == "error":
                raise RuntimeError(
                    f"Worker failed to start: {msg.get('error', 'unknown')}"
                )
        raise TimeoutError("Worker did not become ready within timeout")

    def shutdown(self) -> None:
        """Terminate the worker subprocess gracefully."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return
            try:
                self._send({"cmd": "shutdown"})
                self._proc.wait(timeout=10.0)
            except Exception:
                pass
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
            self._proc = None

    # ------------------------------------------------------------------ #
    # Communication
    # ------------------------------------------------------------------ #

    def infer(self, **kwargs: Any) -> dict[str, Any]:
        """Send an inference request and return the JSON response."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                raise RuntimeError("Worker subprocess is not running")
            self._send({"cmd": "infer", **kwargs})
            return self._recv()

    def _send(self, msg: dict[str, Any]) -> None:
        line = json.dumps(msg, ensure_ascii=False)
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _recv(self) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            stderr = ""
            if self._proc.stderr is not None:
                stderr = self._proc.stderr.read(2000)
            raise RuntimeError(f"Worker closed stdout unexpectedly. stderr: {stderr}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Worker returned non-JSON: {line[:500]}") from exc

    def __del__(self) -> None:
        self.shutdown()

    def __enter__(self) -> SubprocessBackend:
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
