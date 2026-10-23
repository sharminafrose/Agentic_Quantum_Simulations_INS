"""Load the project convention files in ``.claude/rules/`` as LLM context.

The same files guide coding agents (path-scoped rules); here they are given to
the LLM that writes the benchmark interpretation, so its wording follows the
project's rigor rules (units, MPS-level fidelity, T = 0, no absolute intensities).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["load_rule"]

RULES_DIR = Path(__file__).resolve().parent.parent / ".claude" / "rules"


def load_rule(name: str, max_chars: int = 12000) -> str:
    """Return rule ``name`` (e.g. ``"dsf-assembly"``) without YAML frontmatter; empty if missing."""
    path = RULES_DIR / f"{name}.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        text = parts[2] if len(parts) == 3 else text
    return text.strip()[:max_chars]
