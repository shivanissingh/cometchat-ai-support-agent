"""
app/policy/conflict.py — Deterministic conflict detection over retrieved chunks.

Design principles
-----------------
* Operates only on is_authoritative==True chunks (precedence must run first).
* Groups chunks by TOPIC_MAP[chunk.filename]; only same-topic authoritative
  chunks are compared.
* Uses a small registry of per-topic claim extractors because this corpus has
  TWO structurally different conflict types:
    - "returns": numeric day-count conflicts (but TrailPlus 45-day exception
      must NOT be flagged — it is a distinct claim key).
    - "breeze-tumbler-care": categorical dishwasher-safety claims (hand-wash
      body vs all-components-safe); regex for numbers would miss this.
* All other topics have no extractor registered — they are never flagged.
* Returns a ConflictResult with a plain-English explanation suitable for
  passing directly to the LLM orchestrator.

Trap case (must NOT fire)
~~~~~~~~~~~~~~~~~~~~~~~~~
The TrailPlus 45-day window coexists with the current 30-day standard policy.
Both documents are active/official/customer.  The 45-day window is explicitly
membership-conditional ("whose TrailPlus membership was active").  The
extractor assigns it claim key "trailplus_member_days" rather than
"standard_days", so no conflict is registered.

Required case (MUST fire)
~~~~~~~~~~~~~~~~~~~~~~~~~
11-product-care.md says the Breeze Tumbler body "should be hand-washed".
12-breeze-tumbler-product-card.md says "all components are dishwasher safe".
Both are active/official/customer.  The extractor assigns claim key
"body_dishwasher" with values "hand_wash" vs "dishwasher_safe" → conflict.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable

from app.policy.topics import TOPIC_MAP
from app.schemas import Chunk, ConflictResult, RetrievedChunk

# ---------------------------------------------------------------------------
# Type alias for claim extractors
# ---------------------------------------------------------------------------

# An extractor receives a Chunk and returns a dict of {claim_key: claim_value}.
# If it cannot extract any claim it returns {}.
ClaimExtractor = Callable[[Chunk], dict[str, str]]


# ---------------------------------------------------------------------------
# Returns-topic claim extractor
# ---------------------------------------------------------------------------

# Matches patterns like "30 calendar days", "45 days", "7 calendar days",
# and hyphenated forms like "45-calendar-day" used in the TrailPlus doc.
_DAY_COUNT_RE = re.compile(
    r"(\d+)[\s-]*(?:calendar[\s-]+)?days?",
    re.IGNORECASE,
)

# Keywords that identify a TrailPlus membership-conditional context.
_TRAILPLUS_MARKERS = re.compile(
    r"\btrailplus\b|\bmembership\b|\bmember\b", re.IGNORECASE
)


def extract_day_count(chunk: Chunk) -> dict[str, str]:
    """Extract day-count claims from a returns-policy chunk.

    Assigns each extracted number to either "standard_days" (ordinary policy)
    or "trailplus_member_days" (membership-conditional exception) by scanning
    a small context window around the number for TrailPlus/membership markers.

    Returns
    -------
    dict[str, str]
        e.g. {"standard_days": "30"} or {"trailplus_member_days": "45"}
        or both if the chunk mentions both in distinct contexts.
    """
    claims: dict[str, str] = {}
    text = chunk.text

    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        m_list = list(_DAY_COUNT_RE.finditer(sentence))
        if not m_list:
            continue

        is_trailplus_sentence = bool(_TRAILPLUS_MARKERS.search(sentence))

        for m in m_list:
            day_str = m.group(1)
            if is_trailplus_sentence:
                claims["trailplus_member_days"] = day_str
            else:
                if "standard_days" not in claims:
                    claims["standard_days"] = day_str

    return claims


# ---------------------------------------------------------------------------
# Breeze Tumbler care claim extractor
# ---------------------------------------------------------------------------

# Phrases indicating the tumbler body is hand-wash-only.
_HAND_WASH_RE = re.compile(
    r"hand[\s-]?wash(?:ed)?|should be hand[\s-]?wash",
    re.IGNORECASE,
)

# Phrases indicating all components (including the body) are dishwasher-safe.
_DISHWASHER_SAFE_ALL_RE = re.compile(
    r"all\s+components?\s+are\s+dishwasher\s+safe|"
    r"dishwasher\s+safe.*all\s+components?",
    re.IGNORECASE,
)


def extract_dishwasher_claim(chunk: Chunk) -> dict[str, str]:
    """Extract the body-dishwasher-safety claim from a Breeze Tumbler chunk.

    Returns
    -------
    dict[str, str]
        ``{"body_dishwasher": "hand_wash"}`` or
        ``{"body_dishwasher": "dishwasher_safe"}`` or ``{}`` if neither
        pattern matches.
    """
    text = chunk.text
    if _DISHWASHER_SAFE_ALL_RE.search(text):
        return {"body_dishwasher": "dishwasher_safe"}
    if _HAND_WASH_RE.search(text):
        return {"body_dishwasher": "hand_wash"}
    return {}


# ---------------------------------------------------------------------------
# Per-topic extractor registry
# ---------------------------------------------------------------------------

_EXTRACTORS: dict[str, ClaimExtractor] = {
    "returns": extract_day_count,
    "breeze-tumbler-care": extract_dishwasher_claim,
}

# Plain-English conflict explanations, keyed by (topic, claim_key).
_EXPLANATIONS: dict[tuple[str, str], str] = {
    ("returns", "standard_days"): (
        "Our current and another active returns document disagree on the "
        "standard return window — please confirm which day count applies."
    ),
    ("breeze-tumbler-care", "body_dishwasher"): (
        "Our current product-care guide and the product page for this item "
        "disagree about dishwasher safety — recommend hand-washing until we "
        "confirm with the team."
    ),
}


# ---------------------------------------------------------------------------
# ConflictDetector
# ---------------------------------------------------------------------------


class ConflictDetector:
    """Deterministic conflict detector for retrieved authoritative chunks.

    Usage
    -----
    ::

        detector = ConflictDetector()
        result = detector.detect(authoritative_chunks)
    """

    def detect(self, chunks: list[RetrievedChunk]) -> ConflictResult:
        """Run conflict detection and return a ConflictResult.

        Parameters
        ----------
        chunks:
            All retrieved chunks.  Non-authoritative chunks are silently
            ignored — only is_authoritative==True chunks are examined.

        Returns
        -------
        ConflictResult
            ``has_conflict=False`` when no genuine conflict is found.
            ``has_conflict=True`` when two same-topic authoritative chunks
            disagree on the same claim key.
        """
        # Filter to authoritative chunks only.
        auth_chunks = [rc for rc in chunks if rc.is_authoritative]

        # Group by topic.
        by_topic: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for rc in auth_chunks:
            topic = TOPIC_MAP.get(rc.chunk.filename)
            if topic is not None:
                by_topic[topic].append(rc)

        for topic, group in by_topic.items():
            extractor = _EXTRACTORS.get(topic)
            if extractor is None:
                continue  # No extractor for this topic — never flagged.

            # Collect (claim_key → {value: [chunks that said it]}) per topic.
            claim_values: dict[str, dict[str, list[Chunk]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for rc in group:
                for claim_key, claim_val in extractor(rc.chunk).items():
                    claim_values[claim_key][claim_val].append(rc.chunk)

            # A conflict exists when a claim key has more than one distinct value.
            for claim_key, val_map in claim_values.items():
                if len(val_map) > 1:
                    conflicting_chunks = [
                        c for chunks_list in val_map.values() for c in chunks_list
                    ]
                    explanation = _EXPLANATIONS.get(
                        (topic, claim_key),
                        (
                            f"Two active authoritative sources in topic '{topic}' "
                            f"disagree on '{claim_key}' — please confirm with the team."
                        ),
                    )
                    return ConflictResult(
                        has_conflict=True,
                        conflicting_chunks=conflicting_chunks,
                        explanation=explanation,
                    )

        return ConflictResult(has_conflict=False)
