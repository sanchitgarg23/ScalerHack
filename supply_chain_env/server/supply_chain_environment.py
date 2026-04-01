"""
Supply Chain Disruption Management — Core Environment Logic
============================================================

Extends ``openenv.core.env_server.Environment``.

Difficulty tiers
────────────────
  easy   (task 0) — 1 supplier fails, 1 customer order, 1 clear alternative
  medium (task 1) — 3 of 5 suppliers fail, 2 customers, tight budget
  hard   (task 2) — cascading failures, 1 scam supplier, demand spike at step 10

Reward function (partial signals — never binary)
────────────────────────────────────────────────
  • each step:         +0.05  if any customer order progressed toward fulfilment
  • delivery on time:  +0.4   per order fulfilled on time
  • late but notified: +0.2
  • missed:             0.0
  • scam order placed: −0.3   (hard mode only, one-time)
  • wasted query:      −0.1   (querying a supplier already known failed)
  • budget exceeded:   −0.5   and episode ends
  • step limit = 30
"""

from __future__ import annotations

import hashlib
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment

from supply_chain_env.models import (
    CustomerOrder,
    SupplyChainAction,
    SupplyChainObservation,
    SupplyChainState,
    SupplierInfo,
)

# ── Constants ───────────────────────────────────────────────────────────────

MAX_STEPS = 30

# ── Internal dataclasses (not exposed to agent) ────────────────────────────


@dataclass
class PendingDelivery:
    """A purchase in-transit."""

    delivery_id: str
    supplier_name: str
    sku: str
    quantity: int
    cost: float
    arrival_day: int
    is_scam: bool = False            # True → delivery will never arrive


@dataclass
class SupplierState:
    """Server-side supplier record — includes hidden fields the agent can't
    see directly (e.g. ``is_scam``, ``true_status``)."""

    name: str
    true_status: str                 # "active" | "failed" | "delayed" | "scam"
    revealed_status: str             # what the agent has been told so far
    stock: int
    price_per_unit: float
    delivery_days: int
    reliability_score: float
    is_scam: bool = False
    query_count: int = 0             # how many times agent has queried
    last_query_day: int = -1         # day of last query
    last_reported_stock: int = -1    # stock reported on last query (scam contradicts)
    failed_day: int = -1             # day the supplier failed (-1 = still fine)


@dataclass
class OrderState:
    """Server-side customer-order record."""

    order_id: str
    sku: str
    quantity: int
    deadline_day: int
    penalty_per_day: float
    fulfilled: bool = False
    notified: bool = False
    fulfilled_day: int = -1
    units_in_transit: int = 0        # units ordered toward this order


@dataclass
class WorldState:
    """All mutable state for a single episode."""

    episode_id: str = ""
    task_id: int = 0
    day: int = 1
    step_count: int = 0
    budget: float = 0.0
    initial_budget: float = 0.0
    done: bool = False
    total_reward: float = 0.0

    inventory: Dict[str, int] = field(default_factory=dict)
    suppliers: List[SupplierState] = field(default_factory=list)
    orders: List[OrderState] = field(default_factory=list)
    deliveries: List[PendingDelivery] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    disruption_log: List[str] = field(default_factory=list)
    market_conditions: str = "stable"

    known_failed: set = field(default_factory=set)   # supplier names agent KNOWS are failed
    scam_penalty_applied: bool = False                # one-time scam penalty flag


# ── Task presets ────────────────────────────────────────────────────────────


def _build_easy() -> dict:
    """Task 0: 1 supplier fails, 1 customer order, 1 clear alternative."""
    return {
        "budget": 50_000.0,
        "suppliers": [
            SupplierState(
                name="Alpha Manufacturing",
                true_status="failed", revealed_status="unknown",
                stock=0, price_per_unit=80.0, delivery_days=3,
                reliability_score=0.95, failed_day=0,
            ),
            SupplierState(
                name="Bravo Supplies",
                true_status="active", revealed_status="unknown",
                stock=500, price_per_unit=90.0, delivery_days=2,
                reliability_score=0.92,
            ),
            SupplierState(
                name="Charlie Parts",
                true_status="active", revealed_status="unknown",
                stock=300, price_per_unit=100.0, delivery_days=4,
                reliability_score=0.88,
            ),
        ],
        "orders": [
            OrderState(
                order_id="ORD-001", sku="widget-A",
                quantity=100, deadline_day=15, penalty_per_day=200.0,
            ),
        ],
    }


def _build_medium() -> dict:
    """Task 1: 3 of 5 suppliers fail, 2 customers, tight budget."""
    return {
        "budget": 25_000.0,
        "suppliers": [
            SupplierState(
                name="Alpha Manufacturing",
                true_status="failed", revealed_status="unknown",
                stock=0, price_per_unit=85.0, delivery_days=3,
                reliability_score=0.90, failed_day=0,
            ),
            SupplierState(
                name="Bravo Supplies",
                true_status="failed", revealed_status="unknown",
                stock=0, price_per_unit=78.0, delivery_days=4,
                reliability_score=0.85, failed_day=0,
            ),
            SupplierState(
                name="Charlie Parts",
                true_status="active", revealed_status="unknown",
                stock=250, price_per_unit=110.0, delivery_days=3,
                reliability_score=0.80,
            ),
            SupplierState(
                name="Delta Logistics",
                true_status="failed", revealed_status="unknown",
                stock=0, price_per_unit=95.0, delivery_days=2,
                reliability_score=0.75, failed_day=0,
            ),
            SupplierState(
                name="Echo Express",
                true_status="active", revealed_status="unknown",
                stock=400, price_per_unit=105.0, delivery_days=5,
                reliability_score=0.82,
            ),
        ],
        "orders": [
            OrderState(
                order_id="ORD-001", sku="widget-A",
                quantity=150, deadline_day=12, penalty_per_day=350.0,
            ),
            OrderState(
                order_id="ORD-002", sku="widget-B",
                quantity=100, deadline_day=20, penalty_per_day=300.0,
            ),
        ],
    }


def _build_hard() -> dict:
    """Task 2: cascading failures, 1 scam supplier, demand spike at step 10."""
    return {
        "budget": 18_000.0,
        "suppliers": [
            SupplierState(
                name="Alpha Manufacturing",
                true_status="failed", revealed_status="unknown",
                stock=0, price_per_unit=85.0, delivery_days=3,
                reliability_score=0.90, failed_day=0,
            ),
            SupplierState(
                name="Bravo Supplies",
                true_status="delayed", revealed_status="unknown",
                stock=200, price_per_unit=95.0, delivery_days=6,
                reliability_score=0.70,
            ),
            SupplierState(
                name="Charlie Parts",
                true_status="active", revealed_status="unknown",
                stock=180, price_per_unit=115.0, delivery_days=3,
                reliability_score=0.80,
            ),
            SupplierState(
                name="Delta Logistics",
                true_status="active", revealed_status="unknown",
                stock=300, price_per_unit=100.0, delivery_days=4,
                reliability_score=0.78,
            ),
            # Scam supplier — looks active, but delivery never arrives.
            # Detectable: query twice on different days → contradictory stock.
            SupplierState(
                name="Ghost Trading",
                true_status="scam", revealed_status="unknown",
                stock=9999, price_per_unit=45.0, delivery_days=2,
                reliability_score=0.99, is_scam=True,
            ),
            SupplierState(
                name="Foxtrot Global",
                true_status="active", revealed_status="unknown",
                stock=150, price_per_unit=120.0, delivery_days=5,
                reliability_score=0.75,
            ),
        ],
        "orders": [
            OrderState(
                order_id="ORD-001", sku="widget-A",
                quantity=200, deadline_day=10, penalty_per_day=500.0,
            ),
            OrderState(
                order_id="ORD-002", sku="widget-B",
                quantity=120, deadline_day=18, penalty_per_day=400.0,
            ),
            OrderState(
                order_id="ORD-003", sku="widget-A",
                quantity=80, deadline_day=22, penalty_per_day=350.0,
            ),
        ],
    }


TASK_BUILDERS = {0: _build_easy, 1: _build_medium, 2: _build_hard}
TASK_LABELS = {0: "easy", 1: "medium", 2: "hard"}


# ── Environment ─────────────────────────────────────────────────────────────


class SupplyChainEnvironment(Environment):
    """
    OpenEnv RL environment for supply-chain disruption management.

    The agent is a procurement manager who must find alternate suppliers,
    renegotiate contracts, resequence orders, and manage customer commitments
    while budget and time run out.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._w = WorldState()
        self._rng = random.Random()

    # ── reset ───────────────────────────────────────────────────────────

    def reset(self, task_id: int = 0, **kwargs) -> SupplyChainObservation:
        """Start a fresh episode.  ``task_id``: 0=easy, 1=medium, 2=hard."""
        builder = TASK_BUILDERS.get(task_id, TASK_BUILDERS[0])
        preset = builder()

        episode_id = uuid.uuid4().hex[:12]
        # Deterministic seeding from episode_id
        seed = int(hashlib.sha256(episode_id.encode()).hexdigest(), 16) % (2**32)
        self._rng = random.Random(seed)

        self._w = WorldState(
            episode_id=episode_id,
            task_id=task_id,
            day=1,
            step_count=0,
            budget=preset["budget"],
            initial_budget=preset["budget"],
            done=False,
            total_reward=0.0,
            inventory={},
            suppliers=preset["suppliers"],
            orders=preset["orders"],
            deliveries=[],
            alerts=[f"📋 Episode started — difficulty: {TASK_LABELS.get(task_id, 'easy')}"],
            disruption_log=[],
            market_conditions="stable",
            known_failed=set(),
            scam_penalty_applied=False,
        )

        return self._observe("Episode initialised. You are the procurement manager.", done=False)

    # ── step ────────────────────────────────────────────────────────────

    def step(
        self, action: SupplyChainAction, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> SupplyChainObservation:
        """Execute one agent action, advance the world, return results."""
        w = self._w
        if w.done:
            return self._observe("Episode already finished.", done=True)

        w.step_count += 1
        w.alerts = []
        reward = 0.0
        result_msg = ""

        # ── dispatch ────────────────────────────────────────────────────
        atype = action.action_type

        if atype == "query_supplier":
            result_msg, r = self._act_query_supplier(action.target)
            reward += r

        elif atype == "place_order":
            result_msg, r = self._act_place_order(
                action.target, action.quantity, action.price
            )
            reward += r

        elif atype == "negotiate_price":
            result_msg, r = self._act_negotiate(action.target, action.price)
            reward += r

        elif atype == "notify_customer":
            result_msg, r = self._act_notify_customer(action.target, action.message)
            reward += r

        elif atype == "expedite_shipment":
            result_msg, r = self._act_expedite(action.target)
            reward += r

        elif atype == "cancel_order":
            result_msg, r = self._act_cancel(action.target)
            reward += r

        elif atype == "declare_done":
            result_msg, r = self._act_declare_done()
            reward += r

        else:
            result_msg = f"❌ Unknown action type: {atype}"
            reward -= 0.1

        # ── world tick (if episode didn't just end) ─────────────────────
        if not w.done:
            self._process_deliveries()
            self._apply_cascading_events()
            self._check_demand_spike()

            # progress bonus: +0.05 if any order has units moving toward it
            if self._any_order_progressing():
                reward += 0.05

            # advance day
            w.day += 1

            # step limit
            if w.step_count >= MAX_STEPS:
                w.done = True
                w.alerts.append("⏰ Step limit reached (30). Episode over.")

        # ── budget check ────────────────────────────────────────────────
        if w.budget < 0 and not w.done:
            w.done = True
            reward -= 0.5
            w.alerts.append("💸 Budget exceeded! Episode terminated.")

        w.total_reward += reward

        obs = self._observe(result_msg, reward=reward, done=w.done)
        obs.metadata.update({
            "day": w.day,
            "step": w.step_count,
            "action_result": result_msg,
        })
        return obs

    # ── state ───────────────────────────────────────────────────────────

    @property
    def state(self) -> SupplyChainState:
        w = self._w
        return SupplyChainState(
            episode_id=w.episode_id,
            step_count=w.step_count,
            task_id=w.task_id,
            total_reward=round(w.total_reward, 4),
            done=w.done,
            difficulty=TASK_LABELS.get(w.task_id, "easy"),
            disruption_events=list(w.disruption_log),
        )

    # ════════════════════════════════════════════════════════════════════
    #  ACTION HANDLERS
    # ════════════════════════════════════════════════════════════════════

    def _act_query_supplier(self, name: str) -> Tuple[str, float]:
        """
        Reveal a supplier's real status.  Costs 0 budget, uses 1 step.

        Scam detection mechanic: scam supplier reports *contradictory* stock
        levels when queried on different days.

        Wasted query (querying a supplier already known-failed): −0.1.
        """
        w = self._w
        sup = self._find_supplier(name)
        if sup is None:
            return f"❌ Supplier '{name}' not found.", 0.0

        # Wasted query penalty
        if name in w.known_failed:
            return (
                f"⚠️ You already know {name} has failed. Wasted query.",
                -0.1,
            )

        reward = 0.0
        sup.query_count += 1

        # ── Scam supplier logic ─────────────────────────────────────────
        if sup.is_scam:
            # Always appears "active" on query
            sup.revealed_status = "active"
            # But stock contradicts on repeat queries from different days
            if sup.query_count == 1 or sup.last_query_day == w.day:
                reported_stock = self._rng.randint(800, 1200)
            else:
                # Contradictory stock — different from last report
                prev = sup.last_reported_stock
                reported_stock = prev + self._rng.choice([-400, -300, 300, 400])
                reported_stock = max(50, reported_stock)

            sup.last_reported_stock = reported_stock
            sup.last_query_day = w.day

            return (
                f"📦 {sup.name}: status=active, stock={reported_stock}, "
                f"price=${sup.price_per_unit:.2f}/unit, "
                f"delivery={sup.delivery_days} days, "
                f"reliability={sup.reliability_score:.0%}",
                reward,
            )

        # ── Normal supplier ─────────────────────────────────────────────
        sup.revealed_status = sup.true_status
        sup.last_query_day = w.day
        sup.last_reported_stock = sup.stock

        if sup.true_status == "failed":
            w.known_failed.add(name)

        return (
            f"📦 {sup.name}: status={sup.true_status}, stock={sup.stock}, "
            f"price=${sup.price_per_unit:.2f}/unit, "
            f"delivery={sup.delivery_days} days, "
            f"reliability={sup.reliability_score:.0%}",
            reward,
        )

    def _act_place_order(
        self, name: str, quantity: int, offered_price: float
    ) -> Tuple[str, float]:
        """
        Place an order with a supplier.

        Requires: supplier active (by true_status), stock ≥ quantity, budget ≥ cost.
        Scam supplier: order is accepted, budget deducted, but delivery *never* arrives.
        """
        w = self._w
        sup = self._find_supplier(name)
        if sup is None:
            return f"❌ Supplier '{name}' not found.", 0.0

        # ── Scam supplier ───────────────────────────────────────────────
        if sup.is_scam:
            unit_price = offered_price if offered_price > 0 else sup.price_per_unit
            cost = unit_price * quantity
            if cost > w.budget:
                return (
                    f"❌ Insufficient budget (need ${cost:,.0f}, have ${w.budget:,.0f}).",
                    0.0,
                )
            w.budget -= cost
            # Create a delivery that will NEVER arrive
            w.deliveries.append(PendingDelivery(
                delivery_id=uuid.uuid4().hex[:8],
                supplier_name=name,
                sku="widget-A",
                quantity=quantity,
                cost=cost,
                arrival_day=999,  # never
                is_scam=True,
            ))
            # One-time scam penalty
            penalty = 0.0
            if not w.scam_penalty_applied:
                penalty = -0.3
                w.scam_penalty_applied = True
                w.disruption_log.append(
                    f"Day {w.day}: Agent placed order with scam supplier {name}"
                )
            return (
                f"✅ Order placed with {name}: {quantity} units for ${cost:,.0f} "
                f"(ETA: day {w.day + sup.delivery_days}).",
                penalty,
            )

        # ── Failed / delayed supplier ───────────────────────────────────
        if sup.true_status == "failed":
            sup.revealed_status = "failed"
            w.known_failed.add(name)
            return f"❌ {name} has failed — cannot fulfil orders.", 0.0

        if sup.true_status == "delayed":
            # Delayed suppliers can still fulfil but at 2× delivery time
            pass

        # ── Stock check ─────────────────────────────────────────────────
        if quantity <= 0:
            return "❌ Quantity must be positive.", 0.0
        if quantity > sup.stock:
            return (
                f"❌ {name} only has {sup.stock} in stock (requested {quantity}).",
                0.0,
            )

        # ── Budget check ────────────────────────────────────────────────
        unit_price = offered_price if offered_price > 0 else sup.price_per_unit
        cost = unit_price * quantity
        if cost > w.budget:
            return (
                f"❌ Insufficient budget (need ${cost:,.0f}, have ${w.budget:,.0f}).",
                0.0,
            )

        # ── Commit ──────────────────────────────────────────────────────
        w.budget -= cost
        sup.stock -= quantity

        delivery_days = sup.delivery_days
        if sup.true_status == "delayed":
            delivery_days *= 2  # delayed suppliers take twice as long

        # Reliability roll — may add extra days
        if self._rng.random() > sup.reliability_score:
            extra = self._rng.randint(1, 3)
            delivery_days += extra
            w.alerts.append(f"⚠️ Delivery from {name} may be delayed +{extra} days.")

        arrival = w.day + delivery_days
        did = uuid.uuid4().hex[:8]
        w.deliveries.append(PendingDelivery(
            delivery_id=did,
            supplier_name=name,
            sku="widget-A",
            quantity=quantity,
            cost=cost,
            arrival_day=arrival,
        ))

        # Track progress toward orders
        for order in w.orders:
            if not order.fulfilled and order.sku == "widget-A":
                order.units_in_transit += quantity
                break

        return (
            f"✅ Ordered {quantity} units from {name} for ${cost:,.0f} "
            f"(delivery ID: {did}, ETA: day {arrival}).",
            0.0,
        )

    def _act_negotiate(self, name: str, offered_price: float) -> Tuple[str, float]:
        """
        Negotiate a lower price.  40% chance price drops 10-20%.  Uses 1 step.
        """
        w = self._w
        sup = self._find_supplier(name)
        if sup is None:
            return f"❌ Supplier '{name}' not found.", 0.0
        if sup.true_status == "failed":
            return f"❌ {name} has failed — cannot negotiate.", 0.0
        if sup.is_scam:
            # Scam supplier always "agrees" (to lure agent in)
            old = sup.price_per_unit
            sup.price_per_unit = round(offered_price * 0.95, 2)
            return (
                f"🤝 {name} eagerly accepted ${sup.price_per_unit:.2f}/unit "
                f"(was ${old:.2f}). Great deal!",
                0.0,
            )

        # 40% chance of success
        if self._rng.random() < 0.40:
            discount = self._rng.uniform(0.10, 0.20)
            old = sup.price_per_unit
            sup.price_per_unit = round(old * (1 - discount), 2)
            return (
                f"🤝 {name} accepted ${sup.price_per_unit:.2f}/unit "
                f"(was ${old:.2f}, {discount:.0%} discount).",
                0.0,
            )
        else:
            return (
                f"😤 {name} refused to lower price from ${sup.price_per_unit:.2f}/unit.",
                0.0,
            )

    def _act_notify_customer(
        self, order_id: str, message: str
    ) -> Tuple[str, float]:
        """Mark an order as notified — reduces penalty for late delivery."""
        w = self._w
        order = self._find_order(order_id)
        if order is None:
            return f"❌ Order '{order_id}' not found.", 0.0
        if order.notified:
            return f"ℹ️ Customer for {order_id} was already notified.", 0.0

        order.notified = True
        return (
            f"📧 Customer notified about {order_id}: \"{message}\"",
            0.0,
        )

    def _act_expedite(self, target: str) -> Tuple[str, float]:
        """
        Speed up a pending delivery by 2 days.  Costs 30% premium on
        original order cost.
        """
        w = self._w
        for d in w.deliveries:
            if d.supplier_name.lower() == target.lower() or d.delivery_id == target:
                if d.is_scam:
                    # Scam: takes the money, still never arrives
                    premium = round(d.cost * 0.30, 2)
                    if premium > w.budget:
                        return f"❌ Cannot afford expedite fee (${premium:,.0f}).", 0.0
                    w.budget -= premium
                    return (
                        f"🚀 Expedite fee paid for {d.supplier_name}: ${premium:,.0f}. "
                        f"Delivery now ETA day {d.arrival_day - 2}.",
                        0.0,
                    )

                premium = round(d.cost * 0.30, 2)
                if premium > w.budget:
                    return f"❌ Cannot afford expedite fee (${premium:,.0f}).", 0.0

                w.budget -= premium
                d.arrival_day = max(w.day + 1, d.arrival_day - 2)
                return (
                    f"🚀 Expedited delivery from {d.supplier_name} — "
                    f"new ETA: day {d.arrival_day} (cost: ${premium:,.0f}).",
                    0.0,
                )

        return f"❌ No pending delivery matching '{target}' to expedite.", 0.0

    def _act_cancel(self, target: str) -> Tuple[str, float]:
        """Cancel a pending order — 70% budget refund."""
        w = self._w
        for i, d in enumerate(w.deliveries):
            if d.supplier_name.lower() == target.lower() or d.delivery_id == target:
                w.deliveries.pop(i)
                refund = round(d.cost * 0.70, 2)
                w.budget += refund
                # Remove in-transit tracking
                for order in w.orders:
                    if not order.fulfilled:
                        order.units_in_transit = max(
                            0, order.units_in_transit - d.quantity
                        )
                        break
                return (
                    f"🔄 Cancelled order from {d.supplier_name}. "
                    f"Refund: ${refund:,.0f} (70% of ${d.cost:,.0f}).",
                    0.0,
                )

        return f"❌ No pending delivery matching '{target}' to cancel.", 0.0

    def _act_declare_done(self) -> Tuple[str, float]:
        """End episode immediately — trigger final grading."""
        w = self._w
        w.done = True
        reward = 0.0

        for order in w.orders:
            if order.fulfilled:
                if order.fulfilled_day <= order.deadline_day:
                    reward += 0.4   # on time
                elif order.notified:
                    reward += 0.2   # late but notified
                # else: missed → 0.0
            else:
                # unfulfilled → 0.0 (no extra penalty beyond missed reward)
                pass

        fulfilled = sum(1 for o in w.orders if o.fulfilled)
        total = len(w.orders)
        return (
            f"🏁 Episode ended. {fulfilled}/{total} orders fulfilled. "
            f"Budget remaining: ${w.budget:,.0f}.",
            reward,
        )

    # ════════════════════════════════════════════════════════════════════
    #  WORLD SIMULATION
    # ════════════════════════════════════════════════════════════════════

    def _process_deliveries(self) -> None:
        """Check if any pending deliveries have arrived today."""
        w = self._w
        arrived = []
        remaining = []
        for d in w.deliveries:
            if d.is_scam:
                remaining.append(d)  # scam deliveries never arrive
                continue
            if w.day >= d.arrival_day:
                arrived.append(d)
            else:
                remaining.append(d)
        w.deliveries = remaining

        for d in arrived:
            sku = d.sku
            qty = d.quantity
            w.inventory[sku] = w.inventory.get(sku, 0) + qty
            w.alerts.append(
                f"📬 Delivery arrived: {qty} × {sku} from {d.supplier_name}"
            )
            self._try_fulfil_orders(sku)

    def _try_fulfil_orders(self, sku: str) -> None:
        """Auto-fulfil customer orders from current inventory."""
        w = self._w
        for order in w.orders:
            if order.fulfilled:
                continue
            if order.sku == sku and w.inventory.get(sku, 0) >= order.quantity:
                w.inventory[sku] -= order.quantity
                order.fulfilled = True
                order.fulfilled_day = w.day
                w.alerts.append(f"✅ Order {order.order_id} fulfilled on day {w.day}!")

    def _apply_cascading_events(self) -> None:
        """
        Medium & hard mode: additional suppliers may fail as the episode
        progresses, simulating cascading supply-chain disruption.
        """
        w = self._w
        if w.task_id == 0:
            return  # easy mode — no cascading

        # Small chance an active supplier degrades each step
        chance = 0.08 if w.task_id == 1 else 0.12
        active = [s for s in w.suppliers if s.true_status == "active" and not s.is_scam]
        if active and self._rng.random() < chance:
            victim = self._rng.choice(active)
            event = self._rng.choice(["delayed", "failed"])
            victim.true_status = event
            if event == "failed":
                victim.stock = 0
                victim.failed_day = w.day
            msg = f"💥 DISRUPTION: {victim.name} is now {event}!"
            w.alerts.append(msg)
            w.disruption_log.append(f"Day {w.day}: {msg}")

    def _check_demand_spike(self) -> None:
        """Hard mode (task 2): demand spike at step 10 — order quantities increase."""
        w = self._w
        if w.task_id != 2:
            return
        if w.step_count == 10:
            spike_pct = self._rng.uniform(0.3, 0.5)
            for order in w.orders:
                if not order.fulfilled:
                    old_qty = order.quantity
                    order.quantity = int(old_qty * (1 + spike_pct))
                    w.alerts.append(
                        f"📈 DEMAND SPIKE: {order.order_id} quantity increased "
                        f"{old_qty} → {order.quantity}!"
                    )
            w.disruption_log.append(
                f"Day {w.day}: Demand spike — unfulfilled order quantities "
                f"increased by ~{spike_pct:.0%}"
            )

    def _any_order_progressing(self) -> bool:
        """True if any unfulfilled order has units in-transit or in inventory."""
        w = self._w
        for order in w.orders:
            if order.fulfilled:
                continue
            if order.units_in_transit > 0:
                return True
            if w.inventory.get(order.sku, 0) > 0:
                return True
        return False

    # ════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ════════════════════════════════════════════════════════════════════

    def _find_supplier(self, name: str) -> Optional[SupplierState]:
        for s in self._w.suppliers:
            if s.name.lower() == name.lower():
                return s
        return None

    def _find_order(self, order_id: str) -> Optional[OrderState]:
        for o in self._w.orders:
            if o.order_id == order_id:
                return o
        return None

    def _observe(
        self, result: str, reward: Optional[float] = None, done: bool = False
    ) -> SupplyChainObservation:
        """Build the agent-visible observation from internal world state."""
        w = self._w
        # Convert internal SupplierState → agent-visible SupplierInfo
        visible_suppliers = []
        for s in w.suppliers:
            visible_suppliers.append(SupplierInfo(
                name=s.name,
                # Agent only sees what has been revealed (or "unknown")
                status=s.revealed_status if s.revealed_status != "unknown"
                       else "active",  # default assumption
                stock=s.last_reported_stock if s.last_reported_stock >= 0
                      else s.stock,
                price_per_unit=s.price_per_unit,
                delivery_days=s.delivery_days,
                reliability_score=s.reliability_score,
            ))

        visible_orders = []
        for o in w.orders:
            visible_orders.append(CustomerOrder(
                order_id=o.order_id,
                sku=o.sku,
                quantity=o.quantity,
                deadline_day=o.deadline_day,
                fulfilled=o.fulfilled,
                notified=o.notified,
                penalty_per_day=o.penalty_per_day,
            ))

        return SupplyChainObservation(
            day=w.day,
            budget_remaining=round(w.budget, 2),
            inventory=dict(w.inventory),
            suppliers=visible_suppliers,
            customer_orders=visible_orders,
            last_action_result=result,
            alerts=list(w.alerts),
            market_conditions=w.market_conditions,
            reward=reward,
            done=done,
        )
