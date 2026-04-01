"""
Supply Chain Environment — OpenEnv Client
==========================================

Typed client for interacting with the Supply Chain Disruption
Management environment server, built on the OpenEnv EnvClient base.

Two client classes:
  • SupplyChainEnv     — OpenEnv EnvClient (primary, typed generics)
  • SupplyChainWSClient — Async WebSocket client for persistent sessions

Usage (EnvClient — HTTP)
────────────────────────
    from supply_chain_env.client import SupplyChainEnv
    from supply_chain_env.models import SupplyChainAction

    env = SupplyChainEnv(server_url="http://localhost:8000")
    obs = env.reset(task_id=1)
    result = env.step(
        SupplyChainAction(action_type="query_supplier", target="Bravo Supplies")
    )
    print(result.observation, result.reward, result.done)
    state = env.state()
    grade = env.grade(task_id=1)

Usage (WebSocket — persistent session)
──────────────────────────────────────
    import asyncio
    from supply_chain_env.client import SupplyChainWSClient

    async def main():
        async with SupplyChainWSClient("ws://localhost:8000/ws") as client:
            obs = await client.reset(task_id=2)
            obs, reward, done, info = await client.step(action)

    asyncio.run(main())
"""

import requests
from typing import Any, Dict, List, Optional, Tuple
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from supply_chain_env.models import (
    SupplyChainAction,
    SupplyChainObservation,
    SupplyChainState,
)


class SupplyChainEnv(EnvClient[SupplyChainAction, SupplyChainObservation, SupplyChainState]):
    """
    Typed OpenEnv client for the Supply Chain environment.
    Uses WebSocket for persistent sessions and HTTP for custom endpoints (tasks, grade).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        **kwargs,
    ) -> None:
        # EnvClient expects base_url (can be http/ws)
        super().__init__(base_url=base_url, **kwargs)
        self._http_url = base_url.rstrip("/")
        self._episode_log: List[Dict[str, Any]] = []

    # ── OpenEnv required hooks ──────────────────────────────────────────

    def _step_payload(self, action: SupplyChainAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[SupplyChainObservation]:
        obs = SupplyChainObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload["reward"],
            done=payload["done"],
        )

    def _parse_state(self, payload: dict) -> SupplyChainState:
        return SupplyChainState(**payload)

    # ── Overrides for episode logging ───────────────────────────────────

    async def reset(self, **kwargs) -> StepResult[SupplyChainObservation]:
        self._episode_log = []
        result = await super().reset(**kwargs)
        # Record reset observation if needed, though usually logging happens on step
        return result

    async def step(self, action: SupplyChainAction, **kwargs) -> StepResult[SupplyChainObservation]:
        result = await super().step(action, **kwargs)

        # Auto-record for grading
        self._episode_log.append({
            "step": len(self._episode_log) + 1,
            "action": self._step_payload(action),
            "observation": result.observation.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "info": getattr(result.observation, "metadata", {}),
        })

        return result

    # ── Custom Endpoints (HTTP) ────────────────────────────────────────

    def tasks(self) -> List[Dict[str, Any]]:
        """Fetch all available task descriptors from the server."""
        resp = requests.get(f"{self._http_url}/tasks")
        resp.raise_for_status()
        return resp.json()

    def grade(
        self,
        task_id: int,
        episode_log: Optional[List[Dict[str, Any]]] = None,
        final_state: Optional[SupplyChainState] = None,
    ) -> Dict[str, Any]:
        """Run the server-side grader on a completed episode."""
        if episode_log is None:
            episode_log = self._episode_log
        if final_state is None:
            # Note: state() is async, so we might need a sync version or handle it
            # For simplicity in this helper, we assume the user provides it or we're in a sync context
            raise ValueError("final_state must be provided for grading in this context")

        resp = requests.post(
            f"{self._http_url}/grade",
            json={
                "task_id": task_id,
                "episode_log": episode_log,
                "final_state": final_state.model_dump(),
            }
        )
        resp.raise_for_status()
        return resp.json()

    @property
    def episode_log(self) -> List[Dict[str, Any]]:
        return list(self._episode_log)


# ════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET CLIENT (async, persistent sessions)
# ════════════════════════════════════════════════════════════════════════════


class SupplyChainWSClient:
    """
    Async WebSocket client for persistent environment sessions.

    Each connection gets its own isolated environment instance on the server.

    Usage:
        async with SupplyChainWSClient("ws://localhost:8000/ws") as client:
            obs = await client.reset(task_id=0)
            result = await client.step(action)
    """

    def __init__(self, ws_url: str = "ws://localhost:8000/ws") -> None:
        self._ws_url = ws_url
        self._ws = None
        self._session_id: Optional[str] = None
        self._episode_log: List[Dict[str, Any]] = []

    async def connect(self) -> str:
        """Open the WebSocket connection. Returns the session_id."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "Install websockets to use the WS client: pip install websockets"
            )

        self._ws = await websockets.connect(self._ws_url)
        # Server sends session_id on connect
        init_msg = json.loads(await self._ws.recv())
        self._session_id = init_msg["result"]["session_id"]
        return self._session_id

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _send(self, method: str, params: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Send a method call and wait for the response."""
        if self._ws is None:
            raise RuntimeError("Not connected — call connect() first.")

        msg = json.dumps({"method": method, "params": params})
        await self._ws.send(msg)
        raw = await self._ws.recv()
        response = json.loads(raw)

        if response.get("error"):
            raise RuntimeError(f"Server error: {response['error']}")

        return response["result"]

    async def reset(self, task_id: int = 0) -> SupplyChainObservation:
        """Start a new episode."""
        self._episode_log = []
        result = await self._send("reset", {"task_id": task_id})
        return SupplyChainObservation(**result["observation"])

    async def step(
        self, action: SupplyChainAction
    ) -> Tuple[SupplyChainObservation, float, bool, Dict[str, Any]]:
        """Execute one action."""
        params = action.model_dump()
        result = await self._send("step", params)
        obs = SupplyChainObservation(**result["observation"])

        self._episode_log.append({
            "step": len(self._episode_log) + 1,
            "action": params,
            "observation": result["observation"],
            "reward": result["reward"],
            "done": result["done"],
            "info": result.get("info", {}),
        })

        return obs, result["reward"], result["done"], result.get("info", {})

    async def state(self) -> SupplyChainState:
        """Get current episode state."""
        result = await self._send("state")
        return SupplyChainState(**result)

    async def grade(
        self,
        task_id: int,
        episode_log: Optional[List[Dict[str, Any]]] = None,
        final_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run grader over WebSocket."""
        if episode_log is None:
            episode_log = self._episode_log
        if final_state is None:
            st = await self.state()
            final_state = st.model_dump()
        return await self._send("grade", {
            "task_id": task_id,
            "episode_log": episode_log,
            "final_state": final_state,
        })

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def episode_log(self) -> List[Dict[str, Any]]:
        return list(self._episode_log)
