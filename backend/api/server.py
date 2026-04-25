"""
Phase 4 — FastAPI backend exposing the full agentic control stack.

Endpoints:
    GET  /grid_state        - current voltages, loads, solar, time-of-day
    POST /step              - run one RL+OPF control step, return decision log
    POST /operator_command  - parse natural-language command, apply overrides
    GET  /action_log        - history of RL actions, OPF overrides, and commands
    WS   /ws                - push per-step telemetry to the frontend

Run:
    uvicorn backend.api.server:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import os
from collections import deque
from typing import Any, Deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..grid import DER_HOUSES, DER_TYPES, DER_CAPACITY_KW, NUM_DER, NUM_HOUSES
from ..llm import parse_operator_command, predict_event, apply_to_env
from ..rl.env import EnvConfig, SmartGridEnv, V_MAX, V_MIN
from ..safety import verify_action


# ----------------------------------------------------------------------
# Lazy RL agent loading — the server still works without a trained model;
# it then uses zero-curtailment as the "RL proposal" so the OPF layer is
# still exercised end-to-end.
# ----------------------------------------------------------------------
_MODEL_PATH = os.environ.get(
    "SMARTGRID_MODEL",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                 "models", "ppo_smartgrid.zip"),
)


class _AgentWrapper:
    def __init__(self) -> None:
        self._model = None
        self._tried = False

    def _load(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            from stable_baselines3 import PPO
            if os.path.exists(_MODEL_PATH):
                self._model = PPO.load(_MODEL_PATH, device="cpu")
                print(f"[api] loaded PPO model from {_MODEL_PATH}")
            else:
                print(f"[api] no PPO model at {_MODEL_PATH}; using zero-curtailment proposals")
        except Exception as e:  # noqa: BLE001
            print(f"[api] failed to load PPO model: {e}")

    def predict(self, obs: np.ndarray) -> np.ndarray:
        self._load()
        if self._model is None:
            return np.zeros(len(DER_HOUSES), dtype=np.float32)
        action, _ = self._model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)


# ----------------------------------------------------------------------
# App state
# ----------------------------------------------------------------------
class AppState:
    def __init__(self) -> None:
        self.env = SmartGridEnv(EnvConfig(seed=0))
        self.obs, _ = self.env.reset(seed=0)
        self.agent = _AgentWrapper()
        self.action_log: Deque[dict[str, Any]] = deque(maxlen=500)
        self.websockets: list[WebSocket] = []


state = AppState()

app = FastAPI(title="Smart Grid AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _grid_state_payload() -> dict[str, Any]:
    env = state.env
    v = env._last_voltages
    overrides = [
        None if (x != x) else float(x)  # NaN check
        for x in env.manual_overrides.tolist()
    ]
    return {
        "hour": float(env._hour),
        "step": int(env._steps),
        "voltages_pu": [float(x) for x in v],
        "loads_kw": [float(x) for x in env._last_loads_kw],
        "available_solar_kw": [float(x) for x in env._last_available_der_kw],  # legacy
        "available_der_kw": [float(x) for x in env._last_available_der_kw],
        "der_types": list(DER_TYPES),
        "der_capacity_kw": list(DER_CAPACITY_KW),
        "der_house_ids": list(DER_HOUSES),
        "manual_overrides": overrides,
        "house_load_scales": [float(x) for x in env.house_load_scales],
        "house_ids": list(range(1, NUM_HOUSES + 1)),
        "v_min": V_MIN,
        "v_max": V_MAX,
        "violations": [bool(x < V_MIN or x > V_MAX) for x in v],
        "slack_bus": int(env.handles.slack_bus),
        "reward_weights": dict(env.reward_weights),
        "solar_scale": env.cfg.solar_scale,
        "load_scale": env.cfg.load_scale,
    }


async def _broadcast(payload: dict[str, Any]) -> None:
    dead: list[WebSocket] = []
    for ws in state.websockets:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in state.websockets:
            state.websockets.remove(ws)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class OperatorCommand(BaseModel):
    command: str


class EventDescription(BaseModel):
    event_description: str
    apply: bool = True  # if False, only return the prediction without modifying env


class HouseLoadScale(BaseModel):
    house_id: int
    scale: float


class HouseLoadScales(BaseModel):
    scales: list[float]


class DerOverride(BaseModel):
    der_index: int
    # Pin the DER to this delivered power in kW (clamped to nameplate). None clears the pin.
    power_kw: float | None = None
    # Backward-compat: old clients sent `curtailment` in [0,1]; treat as fraction of capacity.
    curtailment: float | None = None


class StepResponse(BaseModel):
    grid_state: dict[str, Any]
    decision: dict[str, Any]
    reward: float


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/grid_state")
def get_grid_state() -> dict[str, Any]:
    return _grid_state_payload()


@app.post("/step", response_model=StepResponse)
async def post_step() -> StepResponse:
    env = state.env

    # 1. RL proposes
    rl_action = state.agent.predict(state.obs)

    # 2. Physics layer verifies
    verdict = verify_action(env, rl_action)
    applied = np.asarray(verdict.applied_curtailment, dtype=np.float32)

    # 3. Apply to live env
    state.obs, reward, term, trunc, info = env.step(applied)
    if term or trunc:
        state.obs, _ = env.reset()

    # 4. Log + broadcast
    entry = {
        "kind": "step",
        "hour": float(info["hour"]),
        "step": int(info["step"]),
        "reward": float(reward),
        "violated": bool(info["violated"]),
        "decision": verdict.to_dict(),
        "available_solar_kw": [float(x) for x in info["available_solar_kw"]],
        "curtailed_kw": [float(x) for x in np.asarray(info.get("curtailed_kw", []), dtype=float).reshape(-1)],
        "voltages_pu": [float(x) for x in info["voltages_pu"]],
    }
    state.action_log.append(entry)
    await _broadcast({"type": "step", "payload": entry,
                      "grid_state": _grid_state_payload()})

    return StepResponse(
        grid_state=_grid_state_payload(),
        decision=verdict.to_dict(),
        reward=float(reward),
    )


@app.post("/operator_command")
async def post_operator_command(cmd: OperatorCommand) -> dict[str, Any]:
    parsed = parse_operator_command(cmd.command)

    env = state.env
    applied: dict[str, Any] = {}
    if "solar_scale" in parsed:
        env.set_scales(solar_scale=parsed["solar_scale"])
        applied["solar_scale"] = env.cfg.solar_scale
    if "load_scale" in parsed:
        env.set_scales(load_scale=parsed["load_scale"])
        applied["load_scale"] = env.cfg.load_scale
    for w in ("voltage_penalty_weight", "curtailment_weight", "stability_bonus"):
        if w in parsed:
            env.set_reward_weights(**{w: parsed[w]})
            applied[w] = env.reward_weights[w]

    entry = {
        "kind": "operator_command",
        "command": cmd.command,
        "parsed": parsed,
        "applied": applied,
    }
    state.action_log.append(entry)
    await _broadcast({"type": "operator_command", "payload": entry,
                      "grid_state": _grid_state_payload()})
    return entry


@app.post("/predict_event")
async def post_predict_event(body: EventDescription) -> dict[str, Any]:
    """LLM Situational Awareness: turn an event description into validated
    grid parameters (and optionally apply them to the live simulation).

    Response includes event_type plus any of solar_scale, wind_scale,
    gas_scale, load_multiplier, voltage_penalty_weight,
    curtailment_penalty_weight (each clamped to its allowed range), plus an
    `applied` field showing which env knobs were actually changed.
    """
    prediction = predict_event(body.event_description)
    applied: dict[str, Any] = {}
    if body.apply and prediction.get("event_type") != "none":
        applied = apply_to_env(state.env, prediction)

    entry = {
        "kind": "predict_event",
        "event_description": body.event_description,
        "prediction": prediction,
        "applied": applied,
    }
    state.action_log.append(entry)
    await _broadcast({"type": "predict_event", "payload": entry,
                      "grid_state": _grid_state_payload()})
    return {**prediction, "applied": applied}


@app.get("/action_log")
def get_action_log(limit: int = 100) -> dict[str, Any]:
    items = list(state.action_log)[-limit:]
    return {"count": len(items), "items": items}


@app.post("/reset")
async def post_reset() -> dict[str, Any]:
    state.obs, _ = state.env.reset(seed=0)
    state.action_log.clear()
    await _broadcast({"type": "reset", "grid_state": _grid_state_payload()})
    return _grid_state_payload()


@app.post("/set_house_load")
async def post_set_house_load(body: HouseLoadScale) -> dict[str, Any]:
    state.env.set_house_load_scale(body.house_id, body.scale)
    entry = {
        "kind": "operator_command",
        "command": f"house {body.house_id} load x{body.scale:.2f}",
        "parsed": {"house_id": body.house_id, "scale": float(body.scale)},
        "applied": {"house_load_scales": [float(x) for x in state.env.house_load_scales]},
    }
    state.action_log.append(entry)
    await _broadcast({"type": "operator_command", "payload": entry,
                      "grid_state": _grid_state_payload()})
    return _grid_state_payload()


@app.post("/set_house_loads")
async def post_set_house_loads(body: HouseLoadScales) -> dict[str, Any]:
    state.env.set_house_load_scales(body.scales)
    entry = {
        "kind": "operator_command",
        "command": "bulk house-load update",
        "parsed": {"scales": [float(x) for x in state.env.house_load_scales]},
        "applied": {"house_load_scales": [float(x) for x in state.env.house_load_scales]},
    }
    state.action_log.append(entry)
    await _broadcast({"type": "operator_command", "payload": entry,
                      "grid_state": _grid_state_payload()})
    return _grid_state_payload()


@app.post("/set_der_override")
async def post_set_der_override(body: DerOverride) -> dict[str, Any]:
    from ..grid import DER_CAPACITY_KW as _CAP

    if body.power_kw is not None:
        pinned: float | None = float(body.power_kw)
    elif body.curtailment is not None:
        # Legacy clients (curtailment fraction in [0,1]) -> equivalent kW vs nameplate.
        cap = float(_CAP[body.der_index])
        pinned = max(0.0, (1.0 - float(body.curtailment)) * cap)
    else:
        pinned = None
    state.env.set_der_override(body.der_index, pinned)
    overrides = [None if (x != x) else float(x) for x in state.env.manual_overrides.tolist()]
    entry = {
        "kind": "operator_command",
        "command": (
            f"clear DER {body.der_index} override"
            if body.curtailment is None
            else f"pin DER {body.der_index} curtailment to {body.curtailment:.2f}"
        ),
        "parsed": {"der_index": body.der_index, "curtailment": body.curtailment},
        "applied": {"manual_overrides": overrides},
    }
    state.action_log.append(entry)
    await _broadcast({"type": "operator_command", "payload": entry,
                      "grid_state": _grid_state_payload()})
    return _grid_state_payload()


@app.post("/clear_der_overrides")
async def post_clear_der_overrides() -> dict[str, Any]:
    state.env.clear_der_overrides()
    entry = {
        "kind": "operator_command",
        "command": "clear all DER overrides",
        "parsed": {},
        "applied": {"manual_overrides": [None] * NUM_DER},
    }
    state.action_log.append(entry)
    await _broadcast({"type": "operator_command", "payload": entry,
                      "grid_state": _grid_state_payload()})
    return _grid_state_payload()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    state.websockets.append(ws)
    try:
        await ws.send_json({"type": "hello", "grid_state": _grid_state_payload()})
        while True:
            # Heartbeat / ignore client messages
            await asyncio.sleep(30)
            try:
                await ws.send_json({"type": "ping"})
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        if ws in state.websockets:
            state.websockets.remove(ws)
