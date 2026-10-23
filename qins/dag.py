"""DAG expansion and a small parallel scheduler.

Agents are expanded into *tasks* (agent x key). A task becomes ready when all
of its upstream tasks finished OK; ready tasks run concurrently in a thread or
process pool. Dependency rule:

* upstream agent has the same fan-out as this agent  -> depend on the task
  with the same key (e.g. ``a03_aqc[xx]`` waits only for ``a02_gs_ansatz[xx]``),
  so different models pipeline independently;
* otherwise -> depend on *all* tasks of the upstream agent.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, BrokenExecutor, Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from threadpoolctl import threadpool_limits

from .agents import AGENT_REGISTRY
from .agents.base import TaskContext, task_id
from .runlog import RunLogger

__all__ = ["Task", "build_tasks", "execute_task", "threads_per_task", "config_hash", "Scheduler"]

_PROCESS_CONFIGURED = False


@dataclass
class Task:
    agent: str
    key: str
    deps: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return task_id(self.agent, self.key)


def _keys_for(fanout: str | None, cfg: dict[str, Any]) -> list[str]:
    if fanout == "model":
        return list(cfg["models"])
    if fanout == "backend":
        return list(cfg["hardware"]["backends"])
    return ["all"]


def build_tasks(cfg: dict[str, Any], agents: list[str] | None = None) -> dict[str, Task]:
    """Expand agents into tasks with explicit task-level dependencies."""
    names = agents or sorted(AGENT_REGISTRY)
    tasks: dict[str, Task] = {}
    for name in names:
        cls = AGENT_REGISTRY[name]
        for key in _keys_for(cls.fanout, cfg):
            deps = []
            for dep in cls.deps:
                if dep not in names:
                    raise ValueError(f"{name} needs {dep}, which is not selected")
                dcls = AGENT_REGISTRY[dep]
                if dcls.fanout == cls.fanout and cls.fanout is not None:
                    deps.append(task_id(dep, key))
                else:
                    deps.extend(task_id(dep, k) for k in _keys_for(dcls.fanout, cfg))
            t = Task(name, key, deps)
            tasks[t.id] = t
    return tasks


def _sparse_warning() -> type[Warning]:
    from scipy.sparse import SparseEfficiencyWarning
    return SparseEfficiencyWarning


def threads_per_task(cfg: dict[str, Any]) -> int:
    """BLAS/Aer threads per task, so parallel tasks do not oversubscribe the CPU."""
    n = cfg["run"].get("threads_per_task")
    return int(n) if n else max(1, (os.cpu_count() or 1) // int(cfg["run"].get("max_workers", 4)))


def config_hash(cfg: dict[str, Any]) -> str:
    """Hash of everything except the ``run`` section (which does not change results)."""
    import hashlib
    body = {k: v for k, v in cfg.items() if k != "run"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _configure_process(cfg: dict[str, Any]) -> int:
    """Process-wide settings (BLAS threads, Aer threads, warnings). Called once per process."""
    threads = threads_per_task(cfg)
    os.environ["QINS_THREADS"] = str(threads)  # read by the Aer simulator
    threadpool_limits(limits=threads)
    warnings.simplefilter("ignore", category=_sparse_warning())  # from PauliEvolutionGate synthesis
    return threads


def execute_task(agent: str, key: str, cfg: dict[str, Any], run_dir: str,
                 upstream: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Run one task (top-level function so it also works in a process pool)."""
    import multiprocessing
    if multiprocessing.parent_process() is not None:  # worker process: configure it once
        global _PROCESS_CONFIGURED
        if not _PROCESS_CONFIGURED:
            _configure_process(cfg)
            _PROCESS_CONFIGURED = True
    ctx = TaskContext(agent, key, cfg, run_dir, upstream)
    t0 = time.time()
    ctx.logger.event("task_start", ctx.task_id, AGENT_REGISTRY[agent].title,
                     seed=ctx.seed, deps=sorted(upstream), threads=threads_per_task(cfg))
    try:
        summary = AGENT_REGISTRY[agent]().run(ctx)
        status, error = "ok", None
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 - incl. Rust PanicException; log, never crash the run
        summary, status = {}, "failed"
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        ctx.logger.event("error", ctx.task_id, str(exc), **error)
    result = {
        "task": ctx.task_id, "agent": agent, "key": key, "status": status,
        "seed": ctx.seed, "duration_s": round(time.time() - t0, 3), "config_hash": config_hash(cfg),
        "upstream_hashes": {d: r.get("artifact_meta", {}).get("result", {}).get("sha256") for d, r in upstream.items()},
        "summary": summary, "error": error,
        "artifacts": {k: v["path"] for k, v in ctx.artifacts.items()},
        "artifact_meta": ctx.artifacts,
    }
    ctx.save_json("result", result)
    ctx.logger.event("task_end", ctx.task_id, status, duration_s=result["duration_s"], summary=summary)
    return result


class Scheduler:
    """Runs the task DAG with maximal parallelism allowed by dependencies."""

    def __init__(self, cfg: dict[str, Any], run_dir: str | Path, tasks: dict[str, Task]) -> None:
        self.cfg, self.run_dir, self.tasks = cfg, str(run_dir), tasks
        self.logger = RunLogger(run_dir)
        self.results: dict[str, dict[str, Any]] = {}

    def _cached(self, tid: str) -> dict[str, Any] | None:
        path = Path(self.run_dir) / "tasks" / tid / "result.json"
        if self.cfg["run"].get("resume") and path.exists():
            res = json.loads(path.read_text())
            if res.get("status") == "ok" and res.get("config_hash") == config_hash(self.cfg):
                return res
            if res.get("status") == "ok":
                self.logger.event("warning", tid, "not reused: config changed since the cached result")
        return None

    def _make_executor(self) -> Executor:
        n = int(self.cfg["run"].get("max_workers", 4))
        if self.cfg["run"].get("executor", "thread") == "process":
            import multiprocessing
            # "spawn": forking a parent that already runs JAX/BLAS threads can deadlock
            return ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"))
        return ThreadPoolExecutor(max_workers=n)

    @staticmethod
    def warm_imports() -> None:
        """Import heavy libraries once in the main thread (concurrent first imports can race)."""
        import qiskit_aer  # noqa: F401
        import qiskit_ibm_runtime  # noqa: F401
        import qiskit_ibm_runtime.fake_provider  # noqa: F401
        import quimb.tensor  # noqa: F401
        import qiskit_quimb  # noqa: F401
        import qiskit_addon_aqc_tensor.simulation.quimb  # noqa: F401
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401

    def run(self) -> dict[str, dict[str, Any]]:
        self.warm_imports()
        _configure_process(self.cfg)  # thread executor: set once here, never from worker threads
        pending = dict(self.tasks)
        running: dict[Future, str] = {}
        with self._make_executor() as pool:
            while pending or running:
                n_before = len(pending)
                for tid in list(pending):
                    task = pending[tid]
                    dep_status = [self.results.get(d, {}).get("status") for d in task.deps]
                    if any(s in ("failed", "skipped") for s in dep_status):
                        pending.pop(tid)
                        self.results[tid] = {"task": tid, "status": "skipped", "summary": {},
                                             "artifacts": {}, "error": {"reason": "upstream failed"}}
                        self.logger.event("task_skip", tid, "upstream failed", deps=task.deps)
                        continue
                    if not all(s == "ok" for s in dep_status):
                        continue
                    pending.pop(tid)
                    cached = self._cached(tid)
                    if cached is not None:
                        self.results[tid] = cached
                        self.logger.event("task_cached", tid, "reused result from previous run")
                        continue
                    upstream = {d: self.results[d] for d in task.deps}
                    try:
                        fut = pool.submit(execute_task, task.agent, task.key, self.cfg, self.run_dir, upstream)
                    except BrokenExecutor as exc:  # a worker died earlier (OOM / segfault)
                        self.results[tid] = {"task": tid, "status": "failed", "summary": {}, "artifacts": {},
                                             "error": {"message": f"executor broken: {exc}"}}
                        self.logger.event("error", tid, f"executor broken: {exc}")
                        continue
                    running[fut] = tid
                    self.logger.event("task_submit", tid, "", deps=task.deps)
                if not running:
                    if pending and len(pending) == n_before:
                        raise RuntimeError(f"DAG is stuck (cycle?): {sorted(pending)}")
                    continue
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in done:
                    tid = running.pop(fut)
                    try:
                        self.results[tid] = fut.result()
                    except BaseException as exc:  # noqa: BLE001 - e.g. a crashed worker process
                        self.results[tid] = {"task": tid, "status": "failed", "summary": {},
                                             "artifacts": {}, "error": {"message": str(exc)}}
                        self.logger.event("error", tid, f"worker crashed: {exc}")
        return self.results
