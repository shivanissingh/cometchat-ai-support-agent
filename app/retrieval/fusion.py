"""
app/retrieval/fusion.py — Reciprocal Rank Fusion (RRF) + metadata scoring.

Pipeline
--------
1. Receive ranked lists from the dense index and the BM25 index.
2. Compute RRF score: sum(1 / (RRF_K + rank_i)) across both rankings.
   A chunk that does not appear in one ranking simply omits that term.
3. Apply metadata bonuses/penalties on top of the RRF score.
4. Return a list of RetrievedChunk sorted by final_score descending.

All tuneable magnitudes are named constants defined near the top of this file
so they can be easily adjusted against the evaluation suite without hunting
through business logic.
"""

from __future__ import annotations

from app.schemas import Chunk, RetrievedChunk

# ---------------------------------------------------------------------------
# RRF constant
# ---------------------------------------------------------------------------

# The standard RRF constant from Cormack, Clarke & Buettcher (2009).
# Do not change this without evaluating against the full evaluation suite.
RRF_K: int = 60

# ---------------------------------------------------------------------------
# Metadata bonus / penalty magnitudes
# (add to / subtract from the raw RRF score)
# ---------------------------------------------------------------------------

BONUS_ACTIVE: float = 0.10          # status == "active"
BONUS_OFFICIAL: float = 0.05        # policy_authority == "official"
BONUS_CUSTOMER: float = 0.05        # audience == "customer"
BONUS_CUSTOMER_ANSWERING: float = 0.05  # customer_answering == True

PENALTY_SUPERSEDED: float = -0.20   # status == "superseded"
PENALTY_DRAFT: float = -0.15        # status == "draft"
PENALTY_INTERNAL_STATUS: float = -0.15  # status == "internal"
PENALTY_INTERNAL_AUDIENCE: float = -0.15  # audience == "internal"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rrf_score(
    dense_ranking: list[tuple[str, float]],
    bm25_ranking: list[tuple[str, float]],
) -> dict[str, float]:
    """Compute per-chunk RRF scores from two ranked lists.

    Ranks are 1-based (rank 1 = highest score).

    Returns
    -------
    dict[str, float]
        Mapping chunk_id → RRF score.
    """
    scores: dict[str, float] = {}

    for rank_0based, (chunk_id, _) in enumerate(dense_ranking):
        rank = rank_0based + 1  # convert to 1-based
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    for rank_0based, (chunk_id, _) in enumerate(bm25_ranking):
        rank = rank_0based + 1
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    return scores


def _metadata_adjustment(chunk: Chunk) -> float:
    """Return the metadata bonus/penalty delta for *chunk*."""
    delta = 0.0

    # Bonuses
    if chunk.status == "active":
        delta += BONUS_ACTIVE
    if chunk.policy_authority == "official":
        delta += BONUS_OFFICIAL
    if chunk.audience == "customer":
        delta += BONUS_CUSTOMER
    if chunk.customer_answering:
        delta += BONUS_CUSTOMER_ANSWERING

    # Penalties
    if chunk.status == "superseded":
        delta += PENALTY_SUPERSEDED
    elif chunk.status == "draft":
        delta += PENALTY_DRAFT
    elif chunk.status == "internal":
        delta += PENALTY_INTERNAL_STATUS

    if chunk.audience == "internal":
        delta += PENALTY_INTERNAL_AUDIENCE

    return delta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    dense_ranking: list[tuple[str, float]],
    bm25_ranking: list[tuple[str, float]],
    chunk_index: dict[str, Chunk],
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Fuse two ranked lists using RRF and apply metadata adjustments.

    Parameters
    ----------
    dense_ranking:
        Ordered (chunk_id, dense_score) pairs from the dense index.
    bm25_ranking:
        Ordered (chunk_id, bm25_score) pairs from the BM25 index.
    chunk_index:
        Mapping chunk_id → Chunk for metadata lookup.
    top_k:
        Maximum number of results to return.

    Returns
    -------
    list[RetrievedChunk]
        Results sorted by ``final_score`` descending.  ``is_authoritative``
        is initialised to ``False`` — call ``app.policy.precedence`` to set it.
    """
    # Build fast score look-ups from the input rankings.
    dense_scores: dict[str, float] = {cid: s for cid, s in dense_ranking}
    bm25_scores: dict[str, float] = {cid: s for cid, s in bm25_ranking}

    rrf_scores = _rrf_score(dense_ranking, bm25_ranking)

    results: list[RetrievedChunk] = []
    for chunk_id, rrf in rrf_scores.items():
        chunk = chunk_index.get(chunk_id)
        if chunk is None:
            continue  # skip unknown ids (should not happen in normal use)
        final = rrf + _metadata_adjustment(chunk)
        results.append(
            RetrievedChunk(
                chunk=chunk,
                dense_score=dense_scores.get(chunk_id, 0.0),
                bm25_score=bm25_scores.get(chunk_id, 0.0),
                rrf_score=rrf,
                final_score=final,
                is_authoritative=False,  # set later by precedence module
            )
        )

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results[:top_k]
