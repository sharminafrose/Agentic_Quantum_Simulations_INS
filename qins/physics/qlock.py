"""One process-wide lock around Qiskit transpilation.

Qiskit's transpiler is not thread-safe (its Rust passes share internal
state; concurrent calls can panic with "RefCell already borrowed"). With the
thread executor, agents therefore transpile one at a time; transpilation is
cheap compared with simulation, so parallelism is barely affected.
"""

from __future__ import annotations

import threading

__all__ = ["TRANSPILE_LOCK"]

TRANSPILE_LOCK = threading.RLock()
