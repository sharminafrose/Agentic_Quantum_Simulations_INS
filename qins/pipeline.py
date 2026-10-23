"""High-level API: build the DAG, run it, write the run report.

    from qins import Pipeline, load_config
    result = Pipeline(load_config("configs/quick.yaml")).run()
    print(result.run_dir, result.status)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .agents import AGENT_REGISTRY
from .config import to_dict
from .dag import Scheduler, Task, build_tasks
from .runlog import RunLogger, environment_info, sha256_file

__all__ = ["Pipeline", "PipelineResult", "mermaid"]


@dataclass
class PipelineResult:
    run_dir: Path
    results: dict[str, dict[str, Any]]
    wall_s: float

    @property
    def status(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results.values():
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out


def mermaid(tasks: dict[str, Task]) -> str:
    """Mermaid flowchart of the task DAG (paste into Obsidian / GitHub)."""
    lines = ["flowchart LR"]
    ids = {tid: f"t{i}" for i, tid in enumerate(tasks)}
    for tid in tasks:
        lines.append(f'  {ids[tid]}["{tid}"]')
    for tid, t in tasks.items():
        lines.extend(f"  {ids[d]} --> {ids[tid]}" for d in t.deps)
    return "\n".join(lines)


def _input_hashes(cfg: dict[str, Any]) -> dict[str, str]:
    files: list[Path] = list(Path(cfg["model_spec"]["papers_dir"]).glob("*.pdf"))
    if cfg["hardware"]["calibration_source"] == "json":
        files += [Path(cfg["hardware"]["calibration_dir"]) / f"{b}.json" for b in cfg["hardware"]["backends"]]
    files += list(Path(cfg["benchmark"]["experiment_dir"]).glob("*.npz"))
    return {str(f): sha256_file(f) for f in files if f.exists()}


class Pipeline:
    def __init__(self, cfg: DictConfig | dict[str, Any]) -> None:
        self.cfg = to_dict(cfg)
        agents = self.cfg["run"].get("agents")
        self.agents = None if agents in (None, "all") else list(agents)
        self.tasks = build_tasks(self.cfg, self.agents)

    def _run_dir(self) -> Path:
        resume = self.cfg["run"].get("resume")
        if resume:
            return Path(resume)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(self.cfg["run"]["out_dir"]) / f"{self.cfg['run']['name']}_{stamp}"

    def run(self, run_dir: str | Path | None = None) -> PipelineResult:
        run_dir = Path(run_dir) if run_dir else self._run_dir()
        logger = RunLogger(run_dir)
        cfg_file = run_dir / "config.yaml"
        if cfg_file.exists():  # resumed run: keep the history of configs
            cfg_file.rename(run_dir / f"config.before_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
        cfg_file.write_text(OmegaConf.to_yaml(OmegaConf.create(self.cfg)))
        (run_dir / "dag.mmd").write_text(mermaid(self.tasks))
        previous = logger.read_manifest()
        history = previous.get("history", []) + ([{k: previous.get(k) for k in ("started", "finished", "status")}]
                                                  if previous else [])
        logger.update_manifest(
            history=history, run_id=run_dir.name, started=datetime.now().isoformat(), config=self.cfg,
            environment=environment_info(), inputs=_input_hashes(self.cfg),
            tasks={tid: {"agent": t.agent, "key": t.key, "deps": t.deps,
                         "title": AGENT_REGISTRY[t.agent].title} for tid, t in self.tasks.items()},
        )
        logger.event("run_start", None, f"{len(self.tasks)} tasks", run_dir=str(run_dir))
        t0 = time.time()
        results = Scheduler(self.cfg, run_dir, self.tasks).run()
        wall = time.time() - t0
        result = PipelineResult(run_dir, results, wall)
        logger.update_manifest(finished=datetime.now().isoformat(), wall_s=round(wall, 2), status=result.status,
                               task_results={tid: {k: r.get(k) for k in ("status", "duration_s", "seed", "artifacts")}
                                             for tid, r in results.items()})
        write_report(run_dir, results, self.cfg, wall)
        logger.event("run_end", None, json.dumps(result.status), wall_s=round(wall, 2))
        return result


def write_report(run_dir: Path, results: dict[str, dict[str, Any]], cfg: dict[str, Any], wall: float) -> None:
    """Collect every a10 metrics table into ``metrics_all.csv`` and a short ``report.md``."""
    import pandas as pd
    rows: list[dict[str, Any]] = []
    for tid, r in results.items():
        if r.get("agent") == "a10_benchmark" and r["status"] == "ok":
            rows += json.loads((run_dir / r["artifacts"]["metrics"]).read_text())
    lines = [f"# Run report: {run_dir.name}", "", f"- wall time: {wall:.1f} s",
             f"- models: {', '.join(cfg['models'])}", f"- backends: {', '.join(cfg['hardware']['backends'])}",
             f"- execution mode: {cfg['execution']['mode']}", "", "## Tasks", "",
             "| task | status | duration (s) |", "|---|---|---|"]
    for tid, r in sorted(results.items()):
        lines.append(f"| {tid} | {r['status']} | {r.get('duration_s', '-')} |")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "metrics_all.csv", index=False)
        cols = [c for c in ("model", "source", "reference", "mse", "wasserstein", "ssim", "nqfi_q_pi", "two_tangle")
                if c in df]
        lines += ["", "## Metrics", "", df[cols].to_markdown(index=False, floatfmt=".4g")]
    notes = [(r["key"], r["summary"]["interpretation"]) for r in results.values()
             if r.get("agent") == "a10_benchmark" and r["status"] == "ok" and (r.get("summary") or {}).get("interpretation")]
    if notes:
        lines += ["", "## LLM interpretation (advisory; check against the tables)", ""]
        for key, n in notes:
            lines += [f"**{key}**: {n['summary']}", ""]
            lines += [f"- caveat: {c}" for c in n.get("caveats", [])] + [f"- suspicious: {c}" for c in n.get("suspicious", [])]
    failed = {tid: r.get("error") for tid, r in results.items() if r["status"] == "failed"}
    if failed:
        lines += ["", "## Failures", ""] + [f"- **{t}**: {(e or {}).get('message')}" for t, e in failed.items()]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
