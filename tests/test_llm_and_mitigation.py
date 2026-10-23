"""LLM client against a mock vLLM server + mitigation policy."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from qins.llm import LLMClient, extract_json
from qins.mitigation import estimator_options, select_mitigation
from qins.runlog import RunLogger

FEATURES = {"n_qubits": 8, "max_n_2q": 300, "max_depth": 500, "max_depth_2q": 90, "idle_fraction": 0.5,
            "est_circuit_fidelity": 0.4, "mean_2q_error": 0.003, "mean_readout_error": 0.01}
MCFG = {"policy": "llm", "force": None, "thresholds": {}}


def _server(reply: str):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"choices": [{"message": {"content": reply}}], "usage": {"total_tokens": 1}})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture()
def llm_cfg():
    return {"model": "mock", "base_url": "", "timeout_s": 5}


def test_extract_json_handles_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json("nonsense") is None


def test_rules_policy():
    out = select_mitigation(FEATURES, {"policy": "rules", "thresholds": {}})
    c = out["choice"]
    assert c["trex"] and c["dd"] and c["twirling"] and c["zne"] and c["dd_sequence"] == "XpXm"
    opts = estimator_options(c, {"shots": 1000})
    assert opts["resilience"]["zne"]["noise_factors"] == [1, 3, 5]


def test_llm_policy_valid_answer(tmp_path, llm_cfg):
    reply = json.dumps({"dd": False, "dd_sequence": "XY4", "twirling": True, "trex": True, "zne": False,
                        "zne_extrapolator": "linear", "reasons": ["short circuit"]})
    srv = _server(reply)
    llm_cfg["base_url"] = f"http://127.0.0.1:{srv.server_port}/v1"
    out = select_mitigation(FEATURES, MCFG, LLMClient(llm_cfg, RunLogger(tmp_path), "t"))
    srv.shutdown()
    assert out["policy_used"] == "llm" and out["choice"]["zne"] is False
    assert (tmp_path / "llm_calls.jsonl").read_text().count("mock") >= 1


def test_llm_policy_invalid_answer_falls_back(tmp_path, llm_cfg):
    srv = _server('{"dd": "maybe"}')
    llm_cfg["base_url"] = f"http://127.0.0.1:{srv.server_port}/v1"
    out = select_mitigation(FEATURES, MCFG, LLMClient(llm_cfg, RunLogger(tmp_path), "t"))
    srv.shutdown()
    assert out["policy_used"] == "rules"


def test_llm_unreachable_falls_back(tmp_path, llm_cfg):
    llm_cfg["base_url"] = "http://127.0.0.1:9/v1"
    out = select_mitigation(FEATURES, MCFG, LLMClient(llm_cfg, RunLogger(tmp_path), "t"))
    assert out["policy_used"] == "rules"


def test_reasoning_prefix_and_sampling_in_request(tmp_path, llm_cfg):
    srv = _server('{"ok": 1}')
    llm_cfg.update(base_url=f"http://127.0.0.1:{srv.server_port}/v1", reasoning="low", top_p=0.95, top_k=64)
    assert LLMClient(llm_cfg, RunLogger(tmp_path), "t").chat_json("sys", "usr") == {"ok": 1}
    srv.shutdown()
    req = json.loads((tmp_path / "llm_calls.jsonl").read_text().splitlines()[0])["request"]
    assert req["messages"][0]["content"].startswith("Reasoning strength: low\n")
    assert req["top_p"] == 0.95 and req["top_k"] == 64


def test_load_rule_strips_frontmatter():
    from qins.rules import load_rule
    text = load_rule("dsf-assembly")
    assert text.startswith("# Neutron-scattering DSF pipeline") and "paths:" not in text[:200]
    assert load_rule("does-not-exist") == ""
