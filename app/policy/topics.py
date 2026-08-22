"""
app/policy/topics.py — Static topic map for the knowledge-base corpus.

This is intentionally a hand-maintained table rather than an auto-derived one.
For a small fixed corpus like this, explicit is more reliable than inferred, and
it is exactly what lets the conflict detector correctly group the returns-policy
trio and the Breeze Tumbler pair even though their document_id prefixes do not
match ("CARE-2026-01" vs "PROD-BREEZE-20").

Maintenance note: when a new file is added to knowledge-base/, add it here.
If it shares a topic with an existing file (i.e. it makes the same factual claims
as another active document), give it the same topic string. Otherwise, use the
filename stem as a unique topic string so that no spurious cross-document conflict
is ever flagged.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Topic map: filename → coarse topic string
# ---------------------------------------------------------------------------
# Files that share a topic are grouped for conflict detection.
# Files with unique topics are never compared against each other.
#
# Returns policy trio — three documents that all contain "N days" claims
# about the return window for Aster & Row orders:
#   01  active/official — 30-day standard + TrailPlus exception (deferred)
#   02  superseded      — legacy 45-day standard (no longer authoritative)
#   09  active/official — TrailPlus 45-day membership-conditional window
#
# Breeze Tumbler pair — two active/official documents that make contradictory
# claims about whether the tumbler body is dishwasher-safe:
#   11  active/official — body must be hand-washed; lid only on top rack
#   12  active/official — all components are dishwasher safe (top rack rec.)
# ---------------------------------------------------------------------------

TOPIC_MAP: dict[str, str] = {
    # --- Returns policy trio ---
    "01-returns-policy-current.md": "returns",
    "02-returns-policy-legacy.md": "returns",
    "09-trailplus-membership.md": "returns",
    # --- Breeze Tumbler care conflict pair ---
    "11-product-care.md": "breeze-tumbler-care",
    "12-breeze-tumbler-product-card.md": "breeze-tumbler-care",
    # --- All other documents: unique topics (filename stem) ---
    "03-final-sale-and-promotions.md": "final-sale-and-promotions",
    "04-damaged-or-wrong-items.md": "damaged-or-wrong-items",
    "05-domestic-shipping.md": "domestic-shipping",
    "06-international-shipping.md": "international-shipping",
    "07-warranty.md": "warranty",
    "08-order-changes-and-cancellations.md": "order-changes-and-cancellations",
    "10-gift-cards-and-price-adjustments.md": "gift-cards-and-price-adjustments",
    "13-support-escalation.md": "support-escalation",
    "14-internal-content-migration-notes.md": "internal-content-migration-notes",
}
