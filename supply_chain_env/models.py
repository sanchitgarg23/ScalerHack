"""
Supply Chain Disruption Management — Typed Models

Defines the core data structures for the RL environment:
  • SupplyChainAction   — the agent's action space
  • SupplyChainObservation — what the agent sees each step
  • SupplyChainState    — episode-level metadata
  • SupplierInfo / CustomerOrder — domain entities
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


# ── Domain entities ─────────────────────────────────────────────────────────


class SupplierInfo(BaseModel):
    """Snapshot of a single supplier's current status."""

    name: str
    status: Literal["active", "failed", "delayed", "scam"]
    stock: int
    price_per_unit: float
    delivery_days: int
    reliability_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Historical reliability — 1.0 = perfect, 0.0 = always fails",
    )


class CustomerOrder(BaseModel):
    """A customer order the agent must fulfil before its deadline."""

    order_id: str
    sku: str
    quantity: int
    deadline_day: int
    fulfilled: bool = False
    notified: bool = False
    penalty_per_day: float = Field(
        default=500.0,
        description="$ penalty for every day the order is late",
    )


# ── Action ──────────────────────────────────────────────────────────────────


class SupplyChainAction(Action):
    """
    An action the procurement-manager agent can take each step.

    Supported action types
    ──────────────────────
    • query_supplier   — inspect a supplier's stock / price / lead-time
    • place_order      — commit budget to buy `quantity` units from `target`
    • negotiate_price  — haggle with `target` for a lower `price`
    • notify_customer  — send an update `message` to the customer of `target` order
    • expedite_shipment— pay extra to rush an in-transit order
    • cancel_order     — cancel a previously placed order (partial refund)
    • declare_done     — signal the agent believes episode is complete
    """

    action_type: Literal[
        "query_supplier",
        "place_order",
        "negotiate_price",
        "notify_customer",
        "expedite_shipment",
        "cancel_order",
        "declare_done",
    ]
    target: str = Field(default="", description="Supplier name or order ID")
    quantity: int = Field(default=0, description="Units to order")
    price: float = Field(default=0.0, description="Offered price per unit")
    message: str = Field(default="", description="Message for notify actions")


# ── Observation ─────────────────────────────────────────────────────────────


class SupplyChainObservation(Observation):
    """
    Everything the agent can see after each step.

    The observation is *partial* — the agent must query suppliers
    to discover up-to-date stock / pricing.
    """

    day: int
    budget_remaining: float
    inventory: Dict[str, int]
    suppliers: List[SupplierInfo]
    customer_orders: List[CustomerOrder]
    last_action_result: str
    alerts: List[str]
    market_conditions: Literal["stable", "volatile", "crisis"] = "stable"


# ── State (episode metadata) ───────────────────────────────────────────────


class SupplyChainState(State):
    """Internal episode state exposed via the /state endpoint."""

    episode_id: str
    step_count: int
    task_id: int
    total_reward: float
    done: bool
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    disruption_events: List[str] = Field(
        default_factory=list,
        description="Log of disruption events that have occurred this episode",
    )
