"""
app/ingestion/parser.py — Parse YAML front-matter and markdown body from
knowledge-base documents.

Each document is parsed into a list of raw section dicts that the chunker
will convert into Chunk objects. The parser itself does NOT split text —
it just extracts structured metadata + the raw body text per file.

Front-matter fields observed across all 14 documents:
  document_id, title, status, effective_date, last_reviewed, audience,
  policy_authority, supersedes, superseded_date, superseded_by,
  customer_answering  (only 14-internal-content-migration-notes.md)

The "topic" field is NOT in any document's front matter; it is assigned via
the static TOPIC_MAP from app.policy.topics.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from app.policy.topics import TOPIC_MAP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent.parent / "knowledge-base"

# Separator between YAML front-matter fences
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """Return (metadata_dict, body_text) for a markdown file with YAML front matter.

    If the file has no front-matter block, returns ({}, raw).
    """
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    meta = yaml.safe_load(m.group(1)) or {}
    body = m.group(2)
    return meta, body


def _derive_customer_answering(meta: dict[str, Any]) -> bool:
    """Determine the customer_answering flag from front-matter metadata.

    Rules:
    - If the document explicitly sets customer_answering=false, return False.
    - If audience is "customer", return True (implicit).
    - Otherwise (audience="internal" without an explicit flag), return False.
    """
    if "customer_answering" in meta:
        # YAML parses 'false' as Python False
        return bool(meta["customer_answering"])
    return meta.get("audience", "internal") == "customer"


def _normalize_status(value: str) -> str:
    """Coerce status to one of the Literal values expected by Chunk."""
    allowed = {"active", "superseded", "draft", "internal"}
    v = str(value).lower().strip()
    return v if v in allowed else "draft"


def _normalize_policy_authority(value: str) -> str:
    """Coerce policy_authority to one of the Literal values expected by Chunk."""
    allowed = {"official", "unofficial", "none"}
    v = str(value).lower().strip()
    return v if v in allowed else "unofficial"


def _normalize_audience(value: str) -> str:
    """Coerce audience to one of the Literal values expected by Chunk."""
    v = str(value).lower().strip()
    return v if v in ("customer", "internal") else "internal"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ParsedDocument:
    """Intermediate container for a parsed knowledge-base document.

    Attributes
    ----------
    filename:
        Basename of the source file (e.g. "01-returns-policy-current.md").
    document_id:
        Value from the ``document_id`` front-matter field.
    title:
        Value from the ``title`` front-matter field.
    status:
        Normalised status literal.
    policy_authority:
        Normalised policy_authority literal.
    audience:
        Normalised audience literal.
    customer_answering:
        Derived customer_answering flag (see ``_derive_customer_answering``).
    effective_date:
        Raw string from front matter, or None.
    last_reviewed:
        Raw string from front matter, or None.
    topic:
        Assigned via TOPIC_MAP (never from front matter).
    body:
        Full markdown body text after the closing ``---`` fence.
    """

    def __init__(self, filename: str, meta: dict[str, Any], body: str) -> None:
        self.filename = filename
        self.document_id: str = str(meta.get("document_id", filename))
        self.title: str = str(meta.get("title", filename))
        self.status: str = _normalize_status(meta.get("status", "draft"))
        self.policy_authority: str = _normalize_policy_authority(
            meta.get("policy_authority", "none")
        )
        self.audience: str = _normalize_audience(meta.get("audience", "internal"))
        self.customer_answering: bool = _derive_customer_answering(meta)
        self.effective_date: str | None = (
            str(meta["effective_date"]) if "effective_date" in meta else None
        )
        self.last_reviewed: str | None = (
            str(meta["last_reviewed"]) if "last_reviewed" in meta else None
        )
        self.topic: str | None = TOPIC_MAP.get(filename)
        self.body: str = body

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ParsedDocument(filename={self.filename!r}, "
            f"document_id={self.document_id!r}, status={self.status!r})"
        )


def parse_file(path: Path) -> ParsedDocument:
    """Parse a single knowledge-base markdown file into a ParsedDocument."""
    raw = path.read_text(encoding="utf-8")
    meta, body = _split_frontmatter(raw)
    return ParsedDocument(filename=path.name, meta=meta, body=body)


def parse_all(knowledge_base_dir: Path = _KNOWLEDGE_BASE_DIR) -> list[ParsedDocument]:
    """Parse every .md file in *knowledge_base_dir* and return them sorted by filename."""
    paths = sorted(knowledge_base_dir.glob("*.md"))
    return [parse_file(p) for p in paths]
