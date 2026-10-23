"""Minimal paper retrieval for agent a00: PDF -> text chunks -> keyword ranking.

Text extraction is cached next to the PDFs (``Papers/.cache/<name>.txt``) so
large libraries are only parsed once. Ranking is a simple term-frequency
score; good enough to put the right paragraphs in front of the LLM.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .runlog import sha256_file

__all__ = ["load_corpus", "top_chunks"]


def _pdf_text(pdf: Path) -> str:
    cache = pdf.parent / ".cache" / f"{pdf.stem}.txt"
    if cache.exists() and cache.stat().st_mtime >= pdf.stat().st_mtime:
        return cache.read_text(encoding="utf-8")
    from pypdf import PdfReader
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def load_corpus(papers_dir: str | Path, chunk_chars: int = 1500) -> list[dict[str, Any]]:
    """Split every PDF in ``papers_dir`` into overlapping chunks with provenance."""
    chunks: list[dict[str, Any]] = []
    for pdf in sorted(Path(papers_dir).glob("*.pdf")):
        text = re.sub(r"\s+", " ", _pdf_text(pdf))
        digest = sha256_file(pdf)[:12]
        step = chunk_chars * 3 // 4
        for k, start in enumerate(range(0, max(len(text), 1), step)):
            chunks.append({"paper": pdf.name, "sha256": digest, "chunk": k,
                           "text": text[start:start + chunk_chars]})
    return chunks


def top_chunks(chunks: list[dict[str, Any]], terms: list[str], k: int = 6) -> list[dict[str, Any]]:
    """Return the ``k`` chunks with most occurrences of the search terms."""
    pats = [re.compile(re.escape(t), re.I) for t in terms if t]
    scored = []
    for c in chunks:
        score = sum(len(p.findall(c["text"])) for p in pats)
        if score > 0:
            scored.append({**c, "score": score})
    return sorted(scored, key=lambda c: -c["score"])[:k]
