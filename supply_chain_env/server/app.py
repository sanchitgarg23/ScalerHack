"""
Supply Chain Environment — FastAPI + WebSocket Server
======================================================

Uses ``openenv.core.env_server.create_app`` to build the standard OpenEnv
HTTP/WS server, then layers on custom endpoints for grading and tasks.

Standard endpoints (from create_app)
────────────────────────────────────
  POST /reset   — {"task_id": 0|1|2} → SupplyChainObservation
  POST /step    — SupplyChainAction   → {observation, reward, done, info}
  GET  /state   — SupplyChainState
  GET  /health  — {"status": "ok"}

Custom endpoints
────────────────
  GET  /tasks   — list of task descriptors
  POST /grade   — {"task_id", "episode_log", "final_state"} → grader score

WebSocket (from create_app)
───────────────────────────
  /ws           — persistent session per connection (isolated env instance)
                  JSON messages: {"method": "reset"|"step"|"state", "params": {...}}
                  Responses:     {"result": {...}, "error": null}

Max 50 concurrent environment sessions.
"""

from __future__ import annotations

from typing import Any, Dict, List

import uvicorn
from fastapi import HTTPException
from pydantic import BaseModel

from openenv.core.env_server import create_app
from supply_chain_env.models import (
    SupplyChainAction,
    SupplyChainObservation,
    SupplyChainState,
)
from supply_chain_env.server.graders import run_grader
from supply_chain_env.server.supply_chain_environment import SupplyChainEnvironment

# ── App setup via OpenEnv create_app ────────────────────────────────────────
# create_app takes a factory function (the lambda), NOT the class directly.

app = create_app(
    lambda: SupplyChainEnvironment(),
    SupplyChainAction,
    SupplyChainObservation,
    max_concurrent_envs=50,
)

# ── Request / response models for custom endpoints ─────────────────────────


class GradeRequest(BaseModel):
    task_id: int
    episode_log: List[Dict[str, Any]]
    final_state: Dict[str, Any]


class TaskInfo(BaseModel):
    task_id: int
    name: str
    difficulty: str
    description: str
    suppliers: int
    customer_orders: int
    budget: float
    step_limit: int


# ════════════════════════════════════════════════════════════════════════════
#  CUSTOM ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════


@app.get("/tasks", response_model=List[TaskInfo])
async def list_tasks():
    """Return metadata for all 3 difficulty tiers."""
    return [
        TaskInfo.model_validate(
            {
                "task_id": 0,
                "name": "Supplier Recovery",
                "difficulty": "easy",
                "description": (
                    "1 supplier fails mid-quarter. Find the alternative, "
                    "fulfil 1 customer order on time. Generous budget."
                ),
                "suppliers": 3,
                "customer_orders": 1,
                "budget": 50_000.0,
                "step_limit": 30,
            }
        ),
        TaskInfo.model_validate(
            {
                "task_id": 1,
                "name": "Multi-Failure Triage",
                "difficulty": "medium",
                "description": (
                    "3 of 5 suppliers have failed. 2 customer orders with tight "
                    "deadlines. Budget is constrained — negotiate wisely."
                ),
                "suppliers": 5,
                "customer_orders": 2,
                "budget": 25_000.0,
                "step_limit": 30,
            }
        ),
        TaskInfo.model_validate(
            {
                "task_id": 2,
                "name": "Cascading Crisis",
                "difficulty": "hard",
                "description": (
                    "Cascading supplier failures, 1 scam supplier disguised as "
                    "active, demand spike at step 10. Razor-thin budget. "
                    "A random agent scores ~0.2–0.35."
                ),
                "suppliers": 6,
                "customer_orders": 3,
                "budget": 18_000.0,
                "step_limit": 30,
            }
        ),
    ]


@app.post("/grade")
async def grade(payload: GradeRequest):
    """
    Run a deterministic grader on a completed episode.

    Body:
        task_id      — 0, 1, or 2
        episode_log  — list of step dicts
        final_state  — SupplyChainState as dict
    """
    try:
        final_state = SupplyChainState(**payload.final_state)
        result = run_grader(
            task_id=payload.task_id,
            episode_log=payload.episode_log,
            final_state=final_state,
        )
        return result
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"No grader registered for task_id={payload.task_id}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    uvicorn.run(
        "supply_chain_env.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
