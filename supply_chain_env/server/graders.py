"""
Supply Chain Environment — Deterministic Episode Graders
=========================================================

Three grader functions that score a completed episode from 0.0 to 1.0.
All graders are **deterministic** — same inputs always produce the same output.

Each grader receives:
    episode_log  — list of dicts, one per step, containing at minimum:
                   {"step": int, "action": SupplyChainAction, "observation": SupplyChainObservation,
                    "reward": float, "done": bool, "info": dict}
    final_state  — SupplyChainState at episode end

Usage:
    from supply_chain_env.server.graders import run_grader
    result = run_grader(task_id=2, episode_log=log, final_state=state)
    # → {"task_id": 2, "score": 0.4725, "max_score": 1.0}
"""

from __future__ import annotations

from typing import Any, Dict, List

from supply_chain_env.models import SupplyChainState


# ════════════════════════════════════════════════════════════════════════════
#  GRADE EASY  (task 0)
# ════════════════════════════════════════════════════════════════════════════


def grade_easy(episode_log: List[Dict[str, Any]], final_state: SupplyChainState) -> float:
    """
    Task 0 — Single supplier fails, 1 customer order.

    Scoring
    ───────
    1.0  — order fulfilled on time
    0.5  — order fulfilled late but customer was notified
    0.2  — order placed but arrived after deadline (no notification)
    0.0  — order never placed or budget exhausted

    Efficiency bonus: +0.0 to +0.1 based on steps used
    (fewer = better, max bonus at ≤ 4 steps).

    Final = min(1.0, base_score + efficiency_bonus)
    """
    if not episode_log:
        return 0.0

    # ── Parse episode for key events ────────────────────────────────────
    order_fulfilled = False
    fulfilled_on_time = False
    customer_notified = False
    order_placed = False
    budget_exhausted = False

    for entry in episode_log:
        action = entry.get("action")
        obs = entry.get("observation")
        info = entry.get("info", {})
        action_result = info.get("action_result", "") or ""

        if action is not None:
            atype = _get_action_type(action)

            if atype == "place_order":
                # Check it was successful (not an error)
                if "✅" in action_result or "Ordered" in action_result:
                    order_placed = True

            elif atype == "notify_customer":
                if "📧" in action_result or "notified" in action_result.lower():
                    customer_notified = True

        # Check observation for fulfilment alerts
        if obs is not None:
            alerts = _get_alerts(obs)
            for alert in alerts:
                if "fulfilled" in alert.lower():
                    order_fulfilled = True

            # Check customer orders in observation
            orders = _get_customer_orders(obs)
            for o in orders:
                if _is_fulfilled(o):
                    order_fulfilled = True

        # Budget exceeded
        if "budget exceeded" in action_result.lower() or "💸" in action_result:
            budget_exhausted = True

    # Also check final observation from last log entry
    if episode_log:
        last_obs = episode_log[-1].get("observation")
        if last_obs is not None:
            orders = _get_customer_orders(last_obs)
            for o in orders:
                if _is_fulfilled(o):
                    order_fulfilled = True
                    # Check if deadline was met
                    deadline = _get_deadline(o)
                    fulfilled_day = _get_fulfilled_day(o, episode_log)
                    if fulfilled_day is not None and deadline is not None:
                        if fulfilled_day <= deadline:
                            fulfilled_on_time = True

    # ── Determine base score ────────────────────────────────────────────
    if budget_exhausted and not order_fulfilled:
        base_score = 0.0
    elif order_fulfilled and fulfilled_on_time:
        base_score = 1.0
    elif order_fulfilled and customer_notified:
        base_score = 0.5
    elif order_fulfilled:
        # Arrived but late, no notification
        base_score = 0.2
    elif order_placed:
        # Placed but never arrived
        base_score = 0.0
    else:
        base_score = 0.0

    # ── Efficiency bonus ────────────────────────────────────────────────
    steps_used = final_state.step_count if final_state else len(episode_log)
    if steps_used <= 4:
        efficiency_bonus = 0.1
    elif steps_used < 15:
        # Linear interpolation: 4 steps → 0.1, 15 steps → 0.0
        efficiency_bonus = 0.1 * max(0.0, (15 - steps_used) / (15 - 4))
    else:
        efficiency_bonus = 0.0

    return min(1.0, base_score + efficiency_bonus)


# ════════════════════════════════════════════════════════════════════════════
#  GRADE MEDIUM  (task 1)
# ════════════════════════════════════════════════════════════════════════════


def grade_medium(episode_log: List[Dict[str, Any]], final_state: SupplyChainState) -> float:
    """
    Task 1 — 3 suppliers fail, 2 customers, budget constraint.

    Component scores (each 0.0–1.0, then weighted):
      fulfilment_rate     — orders fulfilled / total orders        (weight 0.5)
      budget_efficiency   — (start - used) / start                 (weight 0.3)
      negotiation_bonus   — 0.1 per successful negotiation, max 0.2 (weight 0.2)

    Final = 0.5 × fulfilment + 0.3 × budget_efficiency + 0.2 × negotiation_bonus
    """
    if not episode_log:
        return 0.0

    # ── Fulfilment rate ─────────────────────────────────────────────────
    total_orders = 0
    fulfilled_orders = 0

    # Get from final observation
    if episode_log:
        last_obs = episode_log[-1].get("observation")
        if last_obs is not None:
            orders = _get_customer_orders(last_obs)
            total_orders = len(orders)
            fulfilled_orders = sum(1 for o in orders if _is_fulfilled(o))

    # Fallback: count from alerts across all steps
    if total_orders == 0:
        total_orders = 2  # medium has 2 customer orders
        for entry in episode_log:
            obs = entry.get("observation")
            if obs is not None:
                for alert in _get_alerts(obs):
                    if "fulfilled" in alert.lower():
                        fulfilled_orders += 1
        fulfilled_orders = min(fulfilled_orders, total_orders)

    fulfilment_rate = fulfilled_orders / max(total_orders, 1)

    # ── Budget efficiency ───────────────────────────────────────────────
    starting_budget = 25_000.0  # medium task budget
    budget_remaining = 0.0

    if episode_log:
        last_obs = episode_log[-1].get("observation")
        if last_obs is not None:
            budget_remaining = _get_budget(last_obs)

    budget_used = starting_budget - budget_remaining
    budget_efficiency = max(0.0, (starting_budget - budget_used) / starting_budget)

    # ── Negotiation bonus ───────────────────────────────────────────────
    successful_negotiations = 0

    for entry in episode_log:
        action = entry.get("action")
        info = entry.get("info", {})
        action_result = info.get("action_result", "") or ""

        if action is not None and _get_action_type(action) == "negotiate_price":
            if "accepted" in action_result.lower() or "🤝" in action_result:
                successful_negotiations += 1

    # 0.1 per successful negotiation, capped at 0.2
    negotiation_bonus = min(0.2, successful_negotiations * 0.1)

    # Normalise to 0.0–1.0 for weighting (already in range since max is 0.2)
    # But the weight already accounts for the scale, so treat negotiation_bonus
    # as a raw 0.0–1.0 value: 0.2 max → normalise to 1.0
    negotiation_score = min(1.0, negotiation_bonus / 0.2) if negotiation_bonus > 0 else 0.0

    final = (
        0.5 * fulfilment_rate
        + 0.3 * budget_efficiency
        + 0.2 * negotiation_score
    )
    return min(1.0, final)


# ════════════════════════════════════════════════════════════════════════════
#  GRADE HARD  (task 2)
# ════════════════════════════════════════════════════════════════════════════


def grade_hard(episode_log: List[Dict[str, Any]], final_state: SupplyChainState) -> float:
    """
    Task 2 — Cascading failures + scam supplier + demand spike.

    Component scores:
      fulfilment_rate       — orders fulfilled / total        (weight 0.4)
      scam_detected         — 1.0 if agent queried scam supplier twice
                              and did NOT place a second order with it,
                              0.0 otherwise                    (weight 0.25)
      demand_spike_handled  — ratio of post-spike orders
                              fulfilled                        (weight 0.2)
      step_efficiency       — max(0, 1 − steps_used / 30)     (weight 0.15)

    Final = weighted sum, capped at 1.0.

    NOTE: A random LLM agent without RL training should score ~0.2–0.35.
    """
    if not episode_log:
        return 0.0

    # ── Fulfilment rate ─────────────────────────────────────────────────
    total_orders = 0
    fulfilled_orders = 0

    if episode_log:
        last_obs = episode_log[-1].get("observation")
        if last_obs is not None:
            orders = _get_customer_orders(last_obs)
            total_orders = len(orders)
            fulfilled_orders = sum(1 for o in orders if _is_fulfilled(o))

    if total_orders == 0:
        total_orders = 3  # hard has 3 base orders

    fulfilment_rate = fulfilled_orders / max(total_orders, 1)

    # ── Scam detection ──────────────────────────────────────────────────
    scam_supplier_name = "Ghost Trading"
    scam_query_count = 0
    scam_order_count = 0

    for entry in episode_log:
        action = entry.get("action")
        if action is None:
            continue
        atype = _get_action_type(action)
        target = _get_target(action)

        if target.lower() == scam_supplier_name.lower():
            if atype == "query_supplier":
                scam_query_count += 1
            elif atype == "place_order":
                scam_order_count += 1

    # Agent detected scam if: queried ≥ 2 times AND did not place a second order
    if scam_query_count >= 2 and scam_order_count <= 1:
        scam_detected = 1.0
    else:
        scam_detected = 0.0

    # ── Demand spike handled ────────────────────────────────────────────
    # Demand spike happens at step 10 — check if orders that had quantity
    # increased (visible in alerts) were still fulfilled.
    # We track by looking at fulfilment AFTER step 10.
    spike_orders_total = 0
    spike_orders_fulfilled = 0

    # Find orders that existed at step 10 and were unfulfilled (those are spike-affected)
    spike_step_found = False
    spike_affected_order_ids: set = set()

    for entry in episode_log:
        obs = entry.get("observation")
        step_num = entry.get("step", 0)

        if obs is not None:
            alerts = _get_alerts(obs)
            for alert in alerts:
                if "demand spike" in alert.lower() or "DEMAND SPIKE" in alert:
                    spike_step_found = True
                    # Extract order IDs from spike alerts
                    for order_id_candidate in ["ORD-001", "ORD-002", "ORD-003",
                                                "ORD-004", "ORD-005", "ORD-006"]:
                        if order_id_candidate in alert:
                            spike_affected_order_ids.add(order_id_candidate)

    # If spike occurred, check which spike-affected orders were fulfilled
    if spike_step_found and episode_log:
        last_obs = episode_log[-1].get("observation")
        if last_obs is not None:
            orders = _get_customer_orders(last_obs)
            for o in orders:
                oid = _get_order_id(o)
                if oid in spike_affected_order_ids:
                    spike_orders_total += 1
                    if _is_fulfilled(o):
                        spike_orders_fulfilled += 1

    # If no spike orders tracked but spike happened, treat all unfulfilled as spike-affected
    if spike_step_found and spike_orders_total == 0:
        if episode_log:
            last_obs = episode_log[-1].get("observation")
            if last_obs is not None:
                orders = _get_customer_orders(last_obs)
                for o in orders:
                    if not _is_fulfilled(o):
                        spike_orders_total += 1
                    else:
                        spike_orders_total += 1
                        spike_orders_fulfilled += 1

    demand_spike_handled = (
        spike_orders_fulfilled / max(spike_orders_total, 1)
        if spike_orders_total > 0
        else 0.0
    )

    # ── Step efficiency ─────────────────────────────────────────────────
    steps_used = final_state.step_count if final_state else len(episode_log)
    step_efficiency = max(0.0, 1.0 - steps_used / 30.0)

    # ── Final weighted score ────────────────────────────────────────────
    final = (
        0.40 * fulfilment_rate
        + 0.25 * scam_detected
        + 0.20 * demand_spike_handled
        + 0.15 * step_efficiency
    )
    return min(1.0, final)


# ════════════════════════════════════════════════════════════════════════════
#  REGISTRY
# ════════════════════════════════════════════════════════════════════════════


GRADERS = {
    0: grade_easy,
    1: grade_medium,
    2: grade_hard,
}


def run_grader(
    task_id: int,
    episode_log: List[Dict[str, Any]],
    final_state: SupplyChainState,
) -> Dict[str, Any]:
    """
    Run the appropriate grader for a given task.

    Returns:
        {"task_id": int, "score": float (0–1, rounded to 4 dp), "max_score": 1.0}

    Raises:
        KeyError if ``task_id`` has no registered grader.
    """
    grader = GRADERS[task_id]
    score = grader(episode_log, final_state)
    return {
        "task_id": task_id,
        "score": round(score, 4),
        "max_score": 1.0,
    }


# ════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS — extract fields from dicts or Pydantic objects
# ════════════════════════════════════════════════════════════════════════════
#
# Episode log entries may contain either raw dicts or Pydantic model
# instances, so every accessor handles both.


def _get_action_type(action) -> str:
    if isinstance(action, dict):
        return action.get("action_type", "")
    return getattr(action, "action_type", "")


def _get_target(action) -> str:
    if isinstance(action, dict):
        return action.get("target", "")
    return getattr(action, "target", "")


def _get_alerts(obs) -> List[str]:
    if isinstance(obs, dict):
        return obs.get("alerts", [])
    return getattr(obs, "alerts", [])


def _get_customer_orders(obs) -> list:
    if isinstance(obs, dict):
        return obs.get("customer_orders", [])
    return getattr(obs, "customer_orders", [])


def _is_fulfilled(order) -> bool:
    if isinstance(order, dict):
        return order.get("fulfilled", False)
    return getattr(order, "fulfilled", False)


def _get_deadline(order) -> int | None:
    if isinstance(order, dict):
        return order.get("deadline_day")
    return getattr(order, "deadline_day", None)


def _get_order_id(order) -> str:
    if isinstance(order, dict):
        return order.get("order_id", "")
    return getattr(order, "order_id", "")


def _get_budget(obs) -> float:
    if isinstance(obs, dict):
        return obs.get("budget_remaining", 0.0)
    return getattr(obs, "budget_remaining", 0.0)


def _get_fulfilled_day(order, episode_log: list) -> int | None:
    """
    Try to determine the day an order was fulfilled by scanning alerts
    in the episode log for the fulfilment event.
    """
    oid = _get_order_id(order)
    for entry in episode_log:
        obs = entry.get("observation")
        if obs is None:
            continue
        for alert in _get_alerts(obs):
            if oid in alert and "fulfilled" in alert.lower():
                # Try to extract day from "fulfilled on day X"
                parts = alert.lower().split("day")
                if len(parts) >= 2:
                    try:
                        return int(parts[-1].strip().rstrip("!").strip())
                    except ValueError:
                        pass
                # Fallback: use the observation's day
                if isinstance(obs, dict):
                    return obs.get("day")
                return getattr(obs, "day", None)
    return None
