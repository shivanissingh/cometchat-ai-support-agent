"""
app/policy/precedence.py — Mark retrieved chunks as authoritative or not.

A chunk is authoritative only when ALL three conditions hold:
    1. status == "active"
    2. policy_authority == "official"
    3. audience == "customer"

Everything else is retrievable context but must NOT be cited to a customer
as a source of truth, even if it scored highly.

This function MUST be called before conflict detection.  The conflict
detector only inspects is_authoritative==True chunks, which means a
superseded document can never trigger a "conflict" against the current
document — that is precedence, not a genuine contradiction.
"""

from __future__ import annotations

from app.schemas import RetrievedChunk


def mark_authoritative(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Set is_authoritative on each chunk according to the precedence rules.

    The list is modified in-place (a new RetrievedChunk is constructed for
    each element since Pydantic models are immutable by default) and also
    returned for convenience.

    Parameters
    ----------
    chunks:
        List of RetrievedChunk objects, typically returned by
        ``app.retrieval.fusion.reciprocal_rank_fusion``.

    Returns
    -------
    list[RetrievedChunk]
        The same list, with ``is_authoritative`` correctly set on every item.
    """
    updated: list[RetrievedChunk] = []
    for rc in chunks:
        chunk = rc.chunk
        authoritative = (
            chunk.status == "active"
            and chunk.policy_authority == "official"
            and chunk.audience == "customer"
        )
        if authoritative != rc.is_authoritative:
            rc = rc.model_copy(update={"is_authoritative": authoritative})
        updated.append(rc)
    return updated
