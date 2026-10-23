"""Command-line interface.

    qins run  -c configs/quick.yaml -s system.n_sites=10 -s models=[xx,kcuf3]
    qins dag  -c configs/default.yaml          # print the task DAG (mermaid)
    qins agents                                # list agents, fan-out and dependencies
    qins show runs/<run_dir> [--events N]      # status table / tail of the event log
    qins ui                                    # launch the Streamlit app
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

__all__ = ["main"]


def _cmd_run(args: argparse.Namespace) -> int:
    from .config import load_config
    from .pipeline import Pipeline
    overrides = list(args.set)
    if args.resume:
        overrides.append(f"run.resume={args.resume}")
    result = Pipeline(load_config(args.config, overrides)).run(args.run_dir)
    print(f"run dir : {result.run_dir}\nstatus  : {result.status}\nwall    : {result.wall_s:.1f} s")
    print(f"report  : {result.run_dir / 'report.md'}")
    return 0 if "failed" not in result.status else 1


def _cmd_dag(args: argparse.Namespace) -> int:
    from .config import load_config, to_dict
    from .pipeline import Pipeline, mermaid
    print(mermaid(Pipeline(to_dict(load_config(args.config, args.set))).tasks))
    return 0


def _cmd_agents(_: argparse.Namespace) -> int:
    from .agents import AGENT_REGISTRY
    for name, cls in sorted(AGENT_REGISTRY.items()):
        print(f"{name:<20} fanout={str(cls.fanout):<8} deps={list(cls.deps)}\n{'':<20} {cls.title}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    from .runlog import RunLogger
    run = RunLogger(args.run_dir)
    manifest = run.read_manifest()
    for tid, r in sorted((manifest.get("task_results") or {}).items()):
        print(f"{tid:<34} {r['status']:<8} {r.get('duration_s', '')}")
    if args.events:
        for ev in run.read_events()[-args.events:]:
            print(json.dumps({k: ev[k] for k in ("ts", "kind", "task", "msg")}))
    return 0


def _cmd_ui(_: argparse.Namespace) -> int:
    app = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qins", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "dag"):
        sp = sub.add_parser(name)
        sp.add_argument("-c", "--config", default=None, help="YAML merged over configs/default.yaml")
        sp.add_argument("-s", "--set", action="append", default=[], help="dotted override key=value")
        if name == "run":
            sp.add_argument("--resume", default=None, help="run dir to resume (reuses finished tasks)")
            sp.add_argument("--run-dir", default=None, help="explicit output directory for this run")
    sub.add_parser("agents")
    sp = sub.add_parser("show")
    sp.add_argument("run_dir")
    sp.add_argument("--events", type=int, default=0)
    sub.add_parser("ui")
    args = p.parse_args(argv)
    return {"run": _cmd_run, "dag": _cmd_dag, "agents": _cmd_agents, "show": _cmd_show, "ui": _cmd_ui}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
