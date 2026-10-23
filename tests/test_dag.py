"""Scheduler behaviour with tiny dummy agents (no physics)."""

from typing import Any, ClassVar

import pytest

from qins.agents import AGENT_REGISTRY
from qins.agents.base import Agent, TaskContext
from qins.config import load_config, to_dict
from qins.dag import Scheduler, build_tasks


class _Up(Agent):
    name: ClassVar[str] = "t_up"
    fanout: ClassVar[str | None] = "model"

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        if ctx.key == "bad":
            raise RuntimeError("boom")
        ctx.save_json("x", {"key": ctx.key})
        return {"key": ctx.key}


class _Down(Agent):
    name: ClassVar[str] = "t_down"
    fanout: ClassVar[str | None] = "model"
    deps: ClassVar[tuple[str, ...]] = ("t_up",)

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        return {"from_up": ctx.up("t_up")["summary"]["key"]}


class _All(Agent):
    name: ClassVar[str] = "t_all"
    fanout: ClassVar[str | None] = None
    deps: ClassVar[tuple[str, ...]] = ("t_up",)

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        return {"n": len(ctx.up_all("t_up"))}


@pytest.fixture()
def dummy_agents():
    for cls in (_Up, _Down, _All):
        AGENT_REGISTRY[cls.name] = cls
    yield ["t_up", "t_down", "t_all"]
    for cls in (_Up, _Down, _All):
        AGENT_REGISTRY.pop(cls.name)


def _cfg(tmp_path, models):
    cfg = to_dict(load_config(None, [f"run.out_dir={tmp_path}"]))
    cfg["models"] = models
    return cfg


def test_dependency_expansion(dummy_agents, tmp_path):
    tasks = build_tasks(_cfg(tmp_path, ["a", "b"]), dummy_agents)
    assert tasks["t_down[a]"].deps == ["t_up[a]"]
    assert sorted(tasks["t_all[all]"].deps) == ["t_up[a]", "t_up[b]"]


def test_failure_skips_only_dependents(dummy_agents, tmp_path):
    cfg = _cfg(tmp_path, ["good", "bad"])
    res = Scheduler(cfg, tmp_path / "run", build_tasks(cfg, dummy_agents)).run()
    assert res["t_down[good]"]["status"] == "ok" and res["t_down[good]"]["summary"]["from_up"] == "good"
    assert res["t_up[bad]"]["status"] == "failed"
    assert res["t_down[bad]"]["status"] == "skipped"
    assert res["t_all[all]"]["status"] == "skipped"
    assert (tmp_path / "run" / "events.jsonl").exists()


def test_resume_reuses_finished_tasks(dummy_agents, tmp_path):
    cfg = _cfg(tmp_path, ["a"])
    Scheduler(cfg, tmp_path / "run", build_tasks(cfg, dummy_agents)).run()
    cfg["run"]["resume"] = str(tmp_path / "run")
    sched = Scheduler(cfg, tmp_path / "run", build_tasks(cfg, dummy_agents))
    sched.run()
    kinds = [e["kind"] for e in sched.logger.read_events() if e["task"] == "t_up[a]"]
    assert "task_cached" in kinds
    cfg["system"]["n_sites"] = 99  # config change -> must not reuse
    sched = Scheduler(cfg, tmp_path / "run", build_tasks(cfg, dummy_agents))
    sched.run()
    kinds = [e["kind"] for e in sched.logger.read_events() if e["task"] == "t_up[a]"]
    assert kinds.count("task_cached") == 1 and kinds.count("task_start") == 2


def test_real_agent_graph_is_acyclic():
    cfg = to_dict(load_config("quick"))
    tasks = build_tasks(cfg)
    assert len(tasks) == 4 * 9 + 3 + 1  # 9 per-model agents, 3 backends, 1 metrics task
