"""
Pydantic v2 data contracts for the CometChat RAG Support Agent.

Inspection notes
----------------
knowledge-base/ front matter fields observed across all 14 documents:
  document_id, title, status, effective_date, last_reviewed, audience,
  policy_authority, supersedes, superseded_date, superseded_by,
  customer_answering (only in 14-internal-content-migration-notes.md)
  No "topic" field appears in any existing document; it is kept Optional[str]
  as a forward-compatible extension slot.

data/orders.json fields observed per order record:
  order_id, customer (name/email/shipping_address — PII, internal only),
  membership_tier, items (sku/name/quantity/final_sale),
  placed_at, status, status_updated_at, shipped_at, delivered_at,
  carrier, tracking_number, estimated_delivery, customer_safe_message,
  internal (risk_score, warehouse_note, support_tags — internal only)

  NOTE: items.sku is present in the raw data but is NOT in the
  customer-safe whitelist defined by data/orders-data-dictionary.md;
  it is therefore excluded from SafeOrderResult / OrderItem.

  NOTE: There is no "shipping_country" field in the real data; it is
  intentionally absent from SafeOrderResult per the assignment spec.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single document chunk produced during ingestion."""

    chunk_id: str
    filename: str
    document_id: str
    title: str
    heading_path: str
    text: str

    # --- Metadata from knowledge-base front matter ---
    status: Literal["active", "superseded", "draft", "internal"]
    # policy_authority="none" observed on internal/draft docs (e.g. MIG-TEST-04)
    policy_authority: Literal["official", "unofficial", "none"]
    audience: Literal["customer", "internal"]
    # customer_answering observed explicitly only in 14-internal-content-migration-notes.md;
    # all other docs implicitly allow customer answering when audience=customer.
    customer_answering: bool
    effective_date: str | None = None
    last_reviewed: str | None = None
    # topic is not present in any current document; kept as an extension slot.
    topic: str | None = None


class RetrievedChunk(BaseModel):
    """A Chunk annotated with retrieval scores after hybrid search."""

    chunk: Chunk
    dense_score: float
    bm25_score: float
    rrf_score: float
    final_score: float
    is_authoritative: bool


class ConflictResult(BaseModel):
    """Result of a conflict-detection pass over retrieved chunks."""

    has_conflict: bool
    conflicting_chunks: list[Chunk] = Field(default_factory=list)
    explanation: str | None = None


class OrderItem(BaseModel):
    """Customer-safe representation of a single line item.

    Fields correspond to the customer-safe whitelist in
    data/orders-data-dictionary.md: items.name, items.quantity, items.final_sale.
    items.sku is present in the raw data but is NOT on the whitelist and is
    therefore excluded here.
    """

    name: str
    quantity: int
    final_sale: bool


class SafeOrderResult(BaseModel):
    """Customer-safe projection of an order record.

    Field list matches the customer-safe whitelist in
    data/orders-data-dictionary.md exactly. No PII (customer.name,
    customer.email, customer.shipping_address) and no internal fields
    (risk_score, warehouse_note, support_tags) are included.
    There is no shipping_country field in the real data; it is not added here.
    """

    order_id: str
    membership_tier: str | None = None
    items: list[OrderItem] = Field(default_factory=list)
    placed_at: str | None = None
    status: str
    status_updated_at: str | None = None
    shipped_at: str | None = None
    delivered_at: str | None = None
    carrier: str | None = None
    tracking_number: str | None = None
    estimated_delivery: str | None = None
    customer_safe_message: str | None = None
    found: bool
    message: str | None = None


class TraceEvent(BaseModel):
    """A single observability event emitted by any pipeline stage."""

    session_id: str
    turn_id: int
    stage: Literal[
        "router",
        "retrieval",
        "precedence",
        "conflict",
        "tool_call",
        "tool_result",
        "llm_call",
        "validation",
        "response",
    ]
    payload: dict
    timestamp: str
