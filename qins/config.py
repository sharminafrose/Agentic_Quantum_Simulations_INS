"""Configuration loading (YAML + dotted CLI overrides) built on OmegaConf."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf

__all__ = ["CONFIG_DIR", "load_config", "to_dict"]

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> DictConfig:
    """Load ``configs/default.yaml``, merge an optional YAML file, then apply overrides.

    Overrides use dotted keys, e.g. ``["system.n_sites=10", "models=[xx,kcuf3]"]``.
    """
    cfg = OmegaConf.load(CONFIG_DIR / "default.yaml")
    if path is not None:
        p = Path(path)
        for cand in (p, CONFIG_DIR / p, CONFIG_DIR / f"{p}.yaml"):
            if cand.exists():
                p = cand
                break
        cfg = OmegaConf.merge(cfg, OmegaConf.load(p))
    overrides = list(overrides)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def to_dict(cfg: DictConfig | dict) -> dict[str, Any]:
    """Resolve a config into a plain (picklable, JSON-able) dict."""
    if isinstance(cfg, dict):
        return cfg
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
