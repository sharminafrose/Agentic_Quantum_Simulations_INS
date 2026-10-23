#!/usr/bin/env bash
# Start the local LLM server and the Streamlit UI together; the LLM server stops when the UI stops.
#   bash scripts/qins_ui.sh              # Muse-Glimmer on GPU 2 + UI on :8501
#   LLM=0 bash scripts/qins_ui.sh        # UI only
#   GPU=1 PORT=8001 bash scripts/qins_ui.sh
# The server stops on Ctrl-C, when the terminal is closed (VS Code included) or when Streamlit exits.
# Closing only the browser tab does NOT stop Streamlit, so stop it with Ctrl-C or by closing the terminal.
# vLLM needs a few minutes to load; the UI is usable meanwhile (tick "Enable LLM" once /v1/models answers).
set -uo pipefail
cd "$(dirname "$0")/.."

LLM="${LLM:-1}"
LOG="${LLM_LOG:-temp/muse_server.log}"
STREAMLIT="${STREAMLIT:-.venv/bin/streamlit}"
SERVE="${SERVE:-scripts/serve_muse.sh}"
mkdir -p "$(dirname "$LOG")"
VLLM_PID=""

stop_llm() {
  trap - EXIT INT TERM HUP
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[qins_ui] stopping LLM server (process group $VLLM_PID)"
    kill -TERM -- "-$VLLM_PID" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$VLLM_PID" 2>/dev/null
  fi
}
trap stop_llm EXIT INT TERM HUP

if [[ "$LLM" == "1" ]]; then
  if curl -s -m 2 "localhost:${PORT:-8000}/v1/models" >/dev/null 2>&1; then
    echo "[qins_ui] a server already answers on port ${PORT:-8000}; reusing it (it will NOT be stopped)"
  else
    # own session/process group so the whole tree (API server + engine core) can be killed together;
    # pdeathsig stops it even if this script is killed with SIGKILL
    setsid setpriv --pdeathsig TERM bash "$SERVE" > "$LOG" 2>&1 &
    VLLM_PID=$!
    echo "[qins_ui] LLM server starting (log: $LOG)"
  fi
fi

# background + wait so that a signal (terminal closed, kill) runs the trap immediately
"$STREAMLIT" run app/streamlit_app.py --server.fileWatcherType none "$@" &
UI_PID=$!
wait "$UI_PID"
kill -TERM "$UI_PID" 2>/dev/null
