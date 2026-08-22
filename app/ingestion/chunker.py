"""
app/ingestion/chunker.py — Split parsed documents into Chunk objects by
markdown heading structure.

Splitting strategy
------------------
1. Walk the body line-by-line, collecting text under each ## / ### heading.
2. The heading_path for a section under an ## heading is:
       "<document title> > <H2 heading text>"
   For a section under an ### heading beneath an ## heading:
       "<document title> > <H2 heading text> > <H3 heading text>"
3. If a collected section exceeds ~500 tokens (approximated as 375 words),
   split it into multiple sub-chunks of ≤375 words each, all sharing the
   same heading_path (with an incrementing sequence suffix on chunk_id).
4. Sections with no body text after stripping whitespace are skipped.

The heading_path string is shown verbatim as the citation heading in customer
responses, so it must be human-readable.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from app.ingestion.parser import ParsedDocument
from app.schemas import Chunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Approximate word count above which a section is sub-split.
# 500 tokens ≈ 375 words for typical English text.
MAX_WORDS_PER_CHUNK = 375

# Separator used in heading_path strings.
HEADING_SEP = " > "

# Regex patterns for ATX-style markdown headings.
_H1_RE = re.compile(r"^#\s+(.+)$")
_H2_RE = re.compile(r"^##\s+(.+)$")
_H3_RE = re.compile(r"^###\s+(.+)$")


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


class _Section(NamedTuple):
    """Raw section before Chunk construction."""

    heading_path: str
    text: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert arbitrary text to a lowercase, dash-separated identifier fragment."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "-", slug.strip())
    return slug[:60]  # cap length


def _make_chunk_id(document_id: str, heading_path: str, seq: int) -> str:
    """Build a deterministic chunk_id from document metadata."""
    slug = _slugify(heading_path)
    return f"{document_id}-{slug}-{seq}"


def _split_into_sections(title: str, body: str) -> list[_Section]:
    """Walk *body* line-by-line and yield (heading_path, text) pairs.

    The document title (from front matter) is used as the H1 component of
    every heading_path even if the markdown H1 in the body is worded
    differently — the title is what the customer will see.
    """
    sections: list[_Section] = []
    current_h2: str = ""
    current_h3: str = ""
    current_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        if not text:
            current_lines = []
            return
        parts = [title]
        if current_h2:
            parts.append(current_h2)
        if current_h3:
            parts.append(current_h3)
        path = HEADING_SEP.join(parts)
        sections.append(_Section(heading_path=path, text=text))
        current_lines = []

    for line in body.splitlines():
        # Skip the document-level H1 — it is already captured in the title.
        if _H1_RE.match(line):
            continue

        m2 = _H2_RE.match(line)
        if m2:
            _flush()
            current_h2 = m2.group(1).strip()
            current_h3 = ""
            continue

        m3 = _H3_RE.match(line)
        if m3:
            _flush()
            current_h3 = m3.group(1).strip()
            continue

        current_lines.append(line)

    _flush()  # Capture any trailing section.
    return sections


def _word_count(text: str) -> int:
    return len(text.split())


def _sub_split(text: str, max_words: int) -> list[str]:
    """Split *text* into chunks of at most *max_words* words.

    Tries to split on paragraph boundaries first; falls back to word-count
    windowing if a single paragraph exceeds the limit.
    """
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = _word_count(para)
        if current_words + para_words > max_words and current_parts:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = []
            current_words = 0
        if para_words > max_words:
            # Single paragraph is too long — split by words.
            words = para.split()
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i : i + max_words]))
        else:
            current_parts.append(para)
            current_words += para_words

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """Convert a ParsedDocument into a list of Chunk objects.

    Each markdown ## / ### section becomes at least one Chunk. Sections
    exceeding MAX_WORDS_PER_CHUNK are further split while preserving the
    same heading_path.
    """
    sections = _split_into_sections(doc.title, doc.body)
    chunks: list[Chunk] = []

    for section in sections:
        if _word_count(section.text) > MAX_WORDS_PER_CHUNK:
            sub_texts = _sub_split(section.text, MAX_WORDS_PER_CHUNK)
        else:
            sub_texts = [section.text]

        for seq, sub_text in enumerate(sub_texts):
            chunk_id = _make_chunk_id(doc.document_id, section.heading_path, seq)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    filename=doc.filename,
                    document_id=doc.document_id,
                    title=doc.title,
                    heading_path=section.heading_path,
                    text=sub_text,
                    status=doc.status,  # type: ignore[arg-type]
                    policy_authority=doc.policy_authority,  # type: ignore[arg-type]
                    audience=doc.audience,  # type: ignore[arg-type]
                    customer_answering=doc.customer_answering,
                    effective_date=doc.effective_date,
                    last_reviewed=doc.last_reviewed,
                    topic=doc.topic,
                )
            )

    return chunks


def chunk_all_documents(docs: list[ParsedDocument]) -> list[Chunk]:
    """Chunk every document in *docs* and return all chunks in order."""
    result: list[Chunk] = []
    for doc in docs:
        result.extend(chunk_document(doc))
    return result


def build_chunk_index(chunks: list[Chunk]) -> dict[str, Chunk]:
    """Return a dict mapping chunk_id → Chunk for O(1) lookup."""
    return {c.chunk_id: c for c in chunks}
