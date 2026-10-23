"""Thin client for a vLLM server (OpenAI-compatible ``/v1/chat/completions``).

Start a server once and let every agent share it (vLLM batches concurrent
requests, so parallel agents do not wait on each other)::

    bash scripts/serve_muse.sh          # local Muse-Glimmer-30B (fp8, one GPU)
    bash scripts/serve_vllm.sh          # or any other model, e.g. Llama-3.1-8B-Instruct

The LLM is only *advisory*: callers validate the returned JSON and fall back
to deterministic rules when the server is down or the answer is invalid.
Every prompt/response pair is written to ``llm_calls.jsonl``.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from .runlog import RunLogger

__all__ = ["LLMClient", "extract_json"]


def extract_json(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object in ``text`` (tolerates ```json fences)."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


class LLMClient:
    def __init__(self, cfg: dict[str, Any], logger: RunLogger | None = None, task: str | None = None) -> None:
        self.cfg, self.logger, self.task = cfg, logger, task

    def chat(self, system: str, user: str, json_mode: bool = True) -> str:
        reasoning = self.cfg.get("reasoning")  # Muse-Glimmer: "Reasoning strength: <low|medium|high>"
        if reasoning:
            system = f"Reasoning strength: {reasoning}\n{system}"
        payload: dict[str, Any] = {
            "model": self.cfg["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": float(self.cfg.get("temperature", 0.0)),
            "max_tokens": int(self.cfg.get("max_tokens", 1024)),
            "seed": int(self.cfg.get("seed", 0)),
        }
        for key in ("top_p", "top_k"):
            if self.cfg.get(key) is not None:
                payload[key] = self.cfg[key]
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            self.cfg["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.cfg.get('api_key') or 'EMPTY'}"},
            method="POST",
        )
        t0 = time.time()
        record: dict[str, Any] = {"model": self.cfg["model"], "base_url": self.cfg["base_url"],
                                  "request": payload}
        try:
            with urllib.request.urlopen(req, timeout=float(self.cfg.get("timeout_s", 120))) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
            record.update(response=text, usage=data.get("usage"), ok=True)
            return text
        except (urllib.error.URLError, KeyError, IndexError, TimeoutError) as exc:
            record.update(response=None, ok=False, error=str(exc))
            raise ConnectionError(f"LLM call failed: {exc}") from exc
        finally:
            record["latency_s"] = round(time.time() - t0, 3)
            if self.logger is not None:
                self.logger.llm_call(self.task, record)
                self.logger.event("llm_call", self.task, "ok" if record.get("ok") else "failed",
                                  latency_s=record["latency_s"], model=self.cfg["model"])

    def chat_json(self, system: str, user: str) -> dict[str, Any] | None:
        """Return parsed JSON, or ``None`` if the server is unreachable / output invalid."""
        try:
            return extract_json(self.chat(system, user, json_mode=True))
        except ConnectionError:
            return None
