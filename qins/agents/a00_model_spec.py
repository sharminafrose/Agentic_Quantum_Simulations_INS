"""Agent a00 -- material -> Hamiltonian specification.

Starts from the preset / config parameters of each model, retrieves relevant
paragraphs from ``Papers/`` and (if the LLM is enabled) asks the vLLM server
to propose Hamiltonian parameters with citations. The proposal is validated;
it is *applied* only when ``model_spec.accept_llm`` is true, otherwise logged
as a suggestion. The resulting model JSON is the single source of truth for
all downstream agents.
"""

from __future__ import annotations

import json
from typing import Any

from ..papers import load_corpus, top_chunks
from ..physics.models import get_model, model_from_dict
from .base import Agent, TaskContext, register_agent

_FIELDS = {"jxy": (-5.0, 5.0), "jz": (-5.0, 5.0), "hz": (-5.0, 5.0), "hs": (-5.0, 5.0),
           "energy_scale_meV": (1e-3, 1e3)}

_SYSTEM = (
    "You map quantum magnets onto the spin-1/2 chain Hamiltonian "
    "H = J sum_i [jxy (SxSx + SySy) + jz SzSz] + sum_i (hz + (-1)^i hs) Sz_i  (S = sigma/2). "
    "Convention warning: some papers write H with an extra overall factor of 2 (their J is twice ours); "
    "state in notes which convention the source uses and convert to ours. "
    "Use only the provided paper excerpts and well-established literature values. "
    "Reply ONLY with JSON: {\"jxy\": float, \"jz\": float, \"hz\": float, \"hs\": float, "
    "\"energy_scale_meV\": float (=J in meV), \"citations\": [str], \"confidence\": \"low|medium|high\", "
    "\"notes\": str}. Express jxy, jz, hz, hs in units of J (normally jz = 1)."
)


def _validate(raw: dict[str, Any] | None) -> tuple[dict[str, float], list[str]]:
    if raw is None:
        return {}, ["no LLM answer"]
    ok, problems = {}, []
    for key, (lo, hi) in _FIELDS.items():
        try:
            val = float(raw[key])
        except (KeyError, TypeError, ValueError):
            problems.append(f"missing/invalid {key}")
            continue
        if lo <= val <= hi:
            ok[key] = val
        else:
            problems.append(f"{key}={val} outside [{lo}, {hi}]")
    return ok, problems


@register_agent
class ModelSpecAgent(Agent):
    name = "a00_model_spec"
    title = "Material -> Hamiltonian specification"
    fanout = "model"
    deps = ()

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        model = get_model(ctx.key, ctx.cfg)
        spec_cfg = ctx.cfg["model_spec"]
        provenance: dict[str, Any] = {"source": "preset+config", "llm_applied": False}
        evidence: list[dict[str, Any]] = []

        if spec_cfg.get("use_papers", True):
            corpus = load_corpus(spec_cfg["papers_dir"])
            terms = list(model.search_terms) + [model.material]
            evidence = top_chunks(corpus, terms, int(spec_cfg.get("top_k_chunks", 6)))
            ctx.info(f"retrieved {len(evidence)} paper chunks", n_corpus_chunks=len(corpus),
                     papers=sorted({c["paper"] for c in evidence}), terms=terms)
            ctx.save_json("evidence", evidence)

        llm = ctx.llm()
        if llm is not None:
            excerpts = "\n\n".join(f"[{c['paper']} #{c['chunk']}] {c['text']}" for c in evidence) or "(none)"
            user = (f"Material / model: {model.material} ({model.name}). {model.description}\n"
                    f"Current parameters: {json.dumps({k: getattr(model, k) for k in _FIELDS})}\n\n"
                    f"Paper excerpts:\n{excerpts}")
            raw = llm.chat_json(_SYSTEM, user)
            proposal, problems = _validate(raw)
            ctx.save_json("llm_proposal", {"raw": raw, "validated": proposal, "problems": problems})
            apply = bool(spec_cfg.get("accept_llm", False)) and not problems
            ctx.decision("apply_llm_parameters", apply, reasons=problems or ["valid proposal"],
                         proposal=proposal, accept_llm=spec_cfg.get("accept_llm", False))
            if apply:
                model = model_from_dict({**model.to_dict(), **proposal})
                provenance.update(source="llm+papers", llm_applied=True,
                                  citations=(raw or {}).get("citations"), confidence=(raw or {}).get("confidence"))
        else:
            ctx.decision("apply_llm_parameters", False, reasons=["LLM disabled"])

        provenance["evidence_papers"] = sorted({c["paper"] for c in evidence})
        model = model_from_dict({**model.to_dict(), "provenance": provenance})
        ctx.save_json("model", model.to_dict())
        ctx.save_json("hamiltonian_terms", [(p, c.real) for p, c in model.hamiltonian().to_list()])
        return {"model": model.name, "n_sites": model.n_sites, "jxy": model.jxy, "jz": model.jz,
                "hz": model.hz, "hs": model.hs, "J_meV": model.energy_scale_meV, **provenance}
