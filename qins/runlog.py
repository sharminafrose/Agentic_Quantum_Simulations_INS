"""Run-level provenance logging.

Every run writes to ``runs/<run_id>/``::

    manifest.json     config, environment, seeds, input-file hashes, task index
    events.jsonl      one JSON object per event (task start/end, artifacts,
                      decisions, metrics, LLM calls, errors) -- the debug trail
    run.log           the same events as human-readable lines
    llm_calls.jsonl   full prompts and responses of every LLM call
    tasks/<task_id>/  result.json plus all artifacts written by that task

The writer is safe for threads and processes (file append under ``flock``),
so agents running in parallel can all log into the same files.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["RunLogger", "sha256_file", "environment_info", "jsonable"]

_LOCK = threading.Lock()
_TRACKED_PACKAGES = (
    "qins", "numpy", "scipy", "qiskit", "qiskit-aer", "qiskit-ibm-runtime", "quimb",
    "qiskit-addon-aqc-tensor", "qiskit-quimb", "jax", "omegaconf", "streamlit",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def jsonable(obj: Any) -> Any:
    """Convert numpy / path / complex objects into JSON-serialisable values."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist()) if obj.size <= 4096 else f"<ndarray shape={obj.shape}>"
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (complex, np.complexfloating)):
        return {"re": float(obj.real), "im": float(obj.imag)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_info() -> dict[str, Any]:
    """Record interpreter, OS, hardware and package versions for reproducibility."""
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "argv": sys.argv,
        "packages": versions,
    }


class RunLogger:
    """Append-only, process-safe event logger bound to one run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.text_path = self.run_dir / "run.log"
        self.llm_path = self.run_dir / "llm_calls.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"

    # ------------------------------------------------------------------ io
    @staticmethod
    def _append(path: Path, line: str) -> None:
        with _LOCK, open(path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line + "\n")
            f.flush()  # write everything while holding the lock (long lines must not interleave)
            fcntl.flock(f, fcntl.LOCK_UN)

    def event(self, kind: str, task: str | None = None, msg: str = "", **data: Any) -> None:
        """Write one structured event (and a matching human-readable line)."""
        rec = {"ts": _now(), "t_unix": time.time(), "pid": os.getpid(),
               "kind": kind, "task": task, "msg": msg, **jsonable(data)}
        self._append(self.events_path, json.dumps(rec))
        extra = ""
        if data and kind not in ("artifact",):
            extra = " " + json.dumps(jsonable(data))[:400]
        self._append(self.text_path, f"{rec['ts']} [{kind:>9}] {task or '-':<28} {msg}{extra}")

    def llm_call(self, task: str | None, record: dict[str, Any]) -> None:
        self._append(self.llm_path, json.dumps({"ts": _now(), "task": task, **jsonable(record)}))

    # ------------------------------------------------------------ manifest
    def read_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def update_manifest(self, **fields: Any) -> None:
        with _LOCK:
            manifest = self.read_manifest()
            manifest.update(jsonable(fields))
            tmp = self.manifest_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(manifest, indent=2))
            tmp.replace(self.manifest_path)

    # --------------------------------------------------------------- reads
    def read_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        with open(self.events_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
