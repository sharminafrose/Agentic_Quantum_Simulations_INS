"""Streamlit front-end: configure, launch and inspect pipeline runs.

    streamlit run app/streamlit_app.py      (or: qins ui)

Runs are launched as a background ``qins run`` subprocess, so the UI stays
responsive; everything shown here is read back from the run directory
(events.jsonl, manifest.json, task artifacts) -- the same files used for
debugging and for the paper.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qins.config import CONFIG_DIR, load_config, to_dict  # noqa: E402
from qins.dag import build_tasks  # noqa: E402
from qins.physics.models import PRESETS  # noqa: E402
from qins.runlog import RunLogger  # noqa: E402

st.set_page_config(page_title="Quantum INS benchmark", layout="wide")
RUNS = ROOT / "runs"
STATUS_COLORS = {"ok": "🟢", "failed": "🔴", "skipped": "⚪", "running": "🟡", "pending": "⚫", "cached": "🔵"}


# ------------------------------------------------------------------ helpers
def task_status(run_dir: Path) -> pd.DataFrame:
    events = RunLogger(run_dir).read_events()
    manifest = RunLogger(run_dir).read_manifest()
    rows = {tid: {"task": tid, "agent": t["agent"], "key": t["key"], "status": "pending", "start": None,
                  "duration_s": None} for tid, t in (manifest.get("tasks") or {}).items()}
    t0 = min((e["t_unix"] for e in events), default=time.time())
    for e in events:
        tid = e.get("task")
        if tid not in rows:
            continue
        if e["kind"] == "task_start":
            rows[tid].update(status="running", start=round(e["t_unix"] - t0, 1))
        elif e["kind"] == "task_end":
            rows[tid].update(status=e["msg"], duration_s=e.get("duration_s"))
        elif e["kind"] == "task_skip":
            rows[tid]["status"] = "skipped"
        elif e["kind"] == "task_cached":
            rows[tid]["status"] = "cached"
    df = pd.DataFrame(rows.values())
    if not df.empty:
        df.insert(0, "", df["status"].map(STATUS_COLORS))
    return df


def dag_dot(cfg: dict, status: dict[str, str] | None = None) -> str:
    fill = {"ok": "#9be29b", "failed": "#f28b82", "skipped": "#dddddd", "running": "#fdd663", "cached": "#aecbfa"}
    lines = ["digraph G { rankdir=LR; node [shape=box, style=filled, fontsize=10];"]
    for tid, t in build_tasks(cfg).items():
        color = fill.get((status or {}).get(tid, ""), "#ffffff")
        lines.append(f'"{tid}" [fillcolor="{color}"];')
        lines += [f'"{d}" -> "{tid}";' for d in t.deps]
    return "\n".join(lines + ["}"])


def list_runs() -> list[Path]:
    return sorted([p for p in RUNS.glob("*") if (p / "manifest.json").exists()], reverse=True)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Configuration")
configs = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
base = st.sidebar.selectbox("Base config", configs, index=configs.index("quick") if "quick" in configs else 0)
cfg = to_dict(load_config(base))

models = st.sidebar.multiselect("Models", list(PRESETS), default=cfg["models"])
n_sites = st.sidebar.number_input("Chain length (n_sites)", 4, 156, int(cfg["system"]["n_sites"]), step=2)
backends = st.sidebar.multiselect("Backends", ["ibm_kingston", "ibm_boston", "ibm_pittsburgh", "ibm_fez",
                                               "ibm_marrakesh", "ibm_torino"], default=cfg["hardware"]["backends"])
with st.sidebar.expander("Hardware & execution", expanded=False):
    cal_src = st.selectbox("Calibration source", ["fake", "json", "ibm"],
                           index=["fake", "json", "ibm"].index(cfg["hardware"]["calibration_source"]))
    layout = st.selectbox("Layout method", ["chain_search", "qiskit_default", "sangram", "mapomatic"],
                          index=["chain_search", "qiskit_default", "sangram", "mapomatic"].index(cfg["hardware"]["layout_method"]))
    mode = st.selectbox("Execution mode", ["local_emulator", "ibm", "dry_run"],
                        index=["local_emulator", "ibm", "dry_run"].index(cfg["execution"]["mode"]))
    policy = st.selectbox("Mitigation policy", ["rules", "llm"],
                          index=["rules", "llm"].index(cfg["execution"]["mitigation"]["policy"]))
    shots = st.number_input("Shots", 100, 100000, int(cfg["execution"]["shots"]), step=500)
with st.sidebar.expander("LLM (vLLM server)", expanded=False):
    llm_on = st.checkbox("Enable LLM", value=bool(cfg["llm"]["enabled"]))
    llm_url = st.text_input("Base URL", cfg["llm"]["base_url"])
    llm_model = st.text_input("Model", cfg["llm"]["model"])
    accept_llm = st.checkbox("Apply LLM-proposed Hamiltonian parameters", value=bool(cfg["model_spec"]["accept_llm"]))
with st.sidebar.expander("Parallelism", expanded=False):
    executor = st.selectbox("Executor", ["thread", "process"], index=["thread", "process"].index(cfg["run"]["executor"]))
    workers = st.number_input("Max workers", 1, 64, int(cfg["run"]["max_workers"]))
extra = st.sidebar.text_area("Extra overrides (one key=value per line)", "")

overrides = [f"models=[{','.join(models)}]", f"system.n_sites={n_sites}", f"hardware.backends=[{','.join(backends)}]",
             f"hardware.calibration_source={cal_src}", f"hardware.layout_method={layout}", f"execution.mode={mode}",
             f"execution.mitigation.policy={policy}", f"execution.shots={shots}", f"llm.enabled={llm_on}",
             f"llm.base_url={llm_url}", f"llm.model={llm_model}", f"model_spec.accept_llm={accept_llm}",
             f"run.executor={executor}", f"run.max_workers={workers}"] + [x.strip() for x in extra.splitlines() if x.strip()]

if st.sidebar.button("▶ Run pipeline", type="primary", disabled=not models or not backends):
    run_dir = RUNS / f"ui_{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "qins.cli", "run", "-c", base, "--run-dir", str(run_dir)]
    for o in overrides:
        cmd += ["-s", o]
    with open(run_dir / "ui_subprocess.log", "w") as log:
        subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    st.session_state["run_dir"] = str(run_dir)
    st.sidebar.success(f"started {run_dir.name}")

# ------------------------------------------------------------------ main
st.title("Quantum simulation of inelastic neutron scattering — multi-agent benchmark")
runs = list_runs()
names = [p.name for p in runs]
default = Path(st.session_state.get("run_dir", "")).name
sel = st.selectbox("Run", names, index=names.index(default) if default in names else 0) if names else None
tab_mon, tab_dag, tab_res, tab_dec, tab_log, tab_man = st.tabs(
    ["Monitor", "DAG", "Results", "Decisions", "Event log", "Manifest"])

if sel is None:
    tab_dag.graphviz_chart(dag_dot({**cfg, "models": models, "hardware": {**cfg["hardware"], "backends": backends}}))
    st.info("No runs yet - configure on the left and press Run.")
    st.stop()

run_dir = RUNS / sel
status_df = task_status(run_dir)
manifest = RunLogger(run_dir).read_manifest()
run_cfg = manifest.get("config", cfg)

with tab_mon:
    done = "finished" in manifest
    c1, c2, c3 = st.columns(3)
    c1.metric("Tasks", len(status_df))
    c2.metric("Finished", int(status_df["status"].isin(["ok", "failed", "skipped", "cached"]).sum()) if len(status_df) else 0)
    c3.metric("State", "done" if done else "running")
    st.dataframe(status_df, width="stretch", hide_index=True)
    if not done and st.button("Refresh"):
        st.rerun()
    if (run_dir / "report.md").exists():
        st.markdown((run_dir / "report.md").read_text())

with tab_dag:
    st.graphviz_chart(dag_dot(run_cfg, dict(zip(status_df.get("task", []), status_df.get("status", [])))))
    st.caption("Agents fanned out per model / backend; edges are task-level dependencies.")

with tab_res:
    if (run_dir / "metrics_all.csv").exists():
        df = pd.read_csv(run_dir / "metrics_all.csv")
        st.dataframe(df, width="stretch", hide_index=True)
    for m in run_cfg.get("models", []):
        img = run_dir / "tasks" / f"a09_dsf[{m}]" / "dsf_panels.png"
        if img.exists():
            st.subheader(m)
            st.image(str(img))

with tab_dec:
    events = RunLogger(run_dir).read_events()
    dec = [e for e in events if e["kind"] == "decision"]
    for e in dec:
        with st.expander(f"{e['task']} — {e['msg']}: {json.dumps(e.get('choice'))[:120]}"):
            st.json({k: v for k, v in e.items() if k not in ("ts", "t_unix", "pid", "kind")})

with tab_log:
    events = RunLogger(run_dir).read_events()
    kinds = sorted({e["kind"] for e in events})
    pick = st.multiselect("Kinds", kinds, default=[k for k in kinds if k != "artifact"])
    tasks = sorted({e["task"] for e in events if e["task"]})
    tpick = st.multiselect("Tasks", tasks)
    shown = [e for e in events if e["kind"] in pick and (not tpick or e["task"] in tpick)]
    st.dataframe(pd.DataFrame([{k: e.get(k) for k in ("ts", "kind", "task", "msg")} | {"data": json.dumps(
        {k: v for k, v in e.items() if k not in ("ts", "t_unix", "pid", "kind", "task", "msg")})[:300]} for e in shown]),
        width="stretch", hide_index=True)
    if (run_dir / "llm_calls.jsonl").exists():
        st.subheader("LLM calls")
        for line in (run_dir / "llm_calls.jsonl").read_text().splitlines():
            rec = json.loads(line)
            with st.expander(f"{rec['task']} ({rec.get('latency_s')} s, ok={rec.get('ok')})"):
                st.json(rec)

with tab_man:
    st.json(manifest)
