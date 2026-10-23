"""Agent base class, agent registry and the per-task context object.

An *agent* is a small class with

* ``name``    -- unique id, e.g. ``"a01_mps_dmrg"``
* ``fanout``  -- ``"model"`` (one task per spin model), ``"backend"`` (one per
  IBM backend) or ``None`` (a single task)
* ``deps``    -- names of upstream agents whose outputs it needs
* ``run(ctx)``-- does the work and returns a small JSON-able summary dict

Large outputs (arrays, circuits, figures) are written with ``ctx.save_*``;
each save is hashed and logged, so every number in the paper can be traced
back to the task, config and inputs that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, ClassVar

import numpy as np

from ..runlog import RunLogger, jsonable, sha256_file

__all__ = ["Agent", "AGENT_REGISTRY", "register_agent", "TaskContext", "task_id", "load_model"]

AGENT_REGISTRY: dict[str, type["Agent"]] = {}


def register_agent(cls: type["Agent"]) -> type["Agent"]:
    AGENT_REGISTRY[cls.name] = cls
    return cls


def task_id(agent: str, key: str) -> str:
    return f"{agent}[{key}]"


class Agent:
    name: ClassVar[str] = "base"
    title: ClassVar[str] = ""
    fanout: ClassVar[str | None] = "model"
    deps: ClassVar[tuple[str, ...]] = ()

    def run(self, ctx: "TaskContext") -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


class TaskContext:
    """Everything a task needs: config, its key, upstream results and I/O helpers."""

    def __init__(self, agent: str, key: str, cfg: dict[str, Any], run_dir: str | Path,
                 upstream: dict[str, dict[str, Any]]) -> None:
        self.agent, self.key, self.cfg = agent, key, cfg
        self.task_id = task_id(agent, key)
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / "tasks" / self.task_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.upstream = upstream
        self.logger = RunLogger(run_dir)
        self.artifacts: dict[str, dict[str, Any]] = {}
        digest = hashlib.sha256(f"{cfg['run']['seed']}:{self.task_id}".encode()).hexdigest()
        self.seed = int(digest[:8], 16)
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------- logging
    def info(self, msg: str, **data: Any) -> None:
        self.logger.event("log", self.task_id, msg, **data)

    def warning(self, msg: str, **data: Any) -> None:
        self.logger.event("warning", self.task_id, msg, **data)

    def metric(self, name: str, value: Any, **data: Any) -> None:
        self.logger.event("metric", self.task_id, name, value=value, **data)

    def decision(self, name: str, choice: Any, reasons: Any = None, **data: Any) -> None:
        """Log a choice the pipeline made (and why) -- these become Methods text."""
        self.logger.event("decision", self.task_id, name, choice=choice, reasons=reasons, **data)

    # ------------------------------------------------------------ upstream
    def up(self, agent: str, key: str | None = None) -> dict[str, Any]:
        """Return the result of an upstream task (same key by default)."""
        return self.upstream[task_id(agent, key or self.key)]

    def up_all(self, agent: str) -> dict[str, dict[str, Any]]:
        """All results of a fanned-out upstream agent, keyed by task key."""
        prefix = agent + "["
        return {tid[len(prefix):-1]: res for tid, res in self.upstream.items()
                if tid.startswith(prefix)}

    # ---------------------------------------------------------------- save
    def _register(self, name: str, path: Path, **meta: Any) -> str:
        rec = {"path": str(path.relative_to(self.run_dir)), "sha256": sha256_file(path),
               "bytes": path.stat().st_size, **meta}
        self.artifacts[name] = rec
        self.logger.event("artifact", self.task_id, name, **rec)
        return str(path)

    def save_json(self, name: str, obj: Any) -> str:
        path = self.dir / f"{name}.json"
        path.write_text(json.dumps(jsonable(obj), indent=2))
        return self._register(name, path, format="json")

    def save_npz(self, name: str, **arrays: Any) -> str:
        path = self.dir / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        shapes = {k: list(np.shape(v)) for k, v in arrays.items()}
        return self._register(name, path, format="npz", shapes=shapes)

    def save_circuits(self, name: str, circuits: list[Any]) -> str:
        from qiskit import qpy
        path = self.dir / f"{name}.qpy"
        with open(path, "wb") as f:
            qpy.dump(circuits, f)
        return self._register(name, path, format="qpy", n_circuits=len(circuits))

    def save_figure(self, name: str, fig: Any) -> str:
        path = self.dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return self._register(name, path, format="png")

    # ---------------------------------------------------------------- load
    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.run_dir / p

    def load_npz(self, rel: str) -> dict[str, np.ndarray]:
        with np.load(self.path(rel), allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def load_json(self, rel: str) -> Any:
        return json.loads(self.path(rel).read_text())

    def load_circuits(self, rel: str) -> list[Any]:
        from qiskit import qpy
        with open(self.path(rel), "rb") as f:
            return list(qpy.load(f))

    # ----------------------------------------------------------------- llm
    def llm(self) -> Callable[..., Any] | None:
        from ..llm import LLMClient
        llm_cfg = self.cfg.get("llm", {})
        if not llm_cfg.get("enabled", False):
            return None
        return LLMClient(llm_cfg, logger=self.logger, task=self.task_id)


def load_model(ctx: TaskContext, key: str | None = None) -> Any:
    """The authoritative SpinModel for this task, as written by agent a00."""
    from ..physics.models import model_from_dict
    res = ctx.up("a00_model_spec", key)
    return model_from_dict(ctx.load_json(res["artifacts"]["model"]))
