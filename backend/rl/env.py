"""
Phase 1b — Gymnasium wrapper around the pandapower feeder.

Observation (26,) = [
    10 bus voltages (p.u.),
    10 household loads (kW),
     5 available solar (kW, pre-curtailment),
     1 time_of_day in [0, 1),
]

Action (5,) in [0, 1]  — per-DER solar curtailment fraction.
    0.0 => no curtailment (inject full available PV)
    1.0 => full curtailment (inject 0 kW)

Reward:
    * -100 per bus voltage outside [0.95, 1.05] p.u.  (voltage_penalty_weight
      scales the -100 term)
    *  -0.1 * total_curtailed_kW
    *  +10 if every bus is within the safe band
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandapower as pp
from gymnasium import spaces

from ..grid import (
    DER_HOUSES,
    NUM_HOUSES,
    SOLAR_CAPACITY_KW,
    build_grid,
    load_profile,
    set_loads_kw,
    set_solar_kw,
    solar_profile,
)

V_MIN: float = 0.95
V_MAX: float = 1.05
NUM_DER: int = len(DER_HOUSES)
OBS_DIM: int = NUM_HOUSES + NUM_HOUSES + NUM_DER + 1  # 26

# Reward weights (can be overridden at reset by the LLM operator layer)
DEFAULT_REWARD_WEIGHTS = {
    "voltage_penalty_weight": 1.0,   # multiplies the -100 per-violation term
    "curtailment_weight": 0.1,       # -w * total_curtailed_kW
    "stability_bonus": 10.0,         # +bonus when all-in-band
}


@dataclass
class EnvConfig:
    dt_hours: float = 1.0         # one step = 1 hour
    episode_hours: float = 24.0   # one episode = one day
    start_hour: float = 0.0
    solar_scale: float = 1.0      # LLM can shrink/grow available PV
    load_scale: float = 1.0
    seed: int | None = None


class SmartGridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.reward_weights = dict(DEFAULT_REWARD_WEIGHTS)

        self.observation_space = spaces.Box(
            low=np.concatenate([
                np.full(NUM_HOUSES, 0.80, dtype=np.float32),           # voltages
                np.full(NUM_HOUSES, 0.0, dtype=np.float32),            # loads
                np.full(NUM_DER, 0.0, dtype=np.float32),               # solar
                np.array([0.0], dtype=np.float32),                     # tod
            ]),
            high=np.concatenate([
                np.full(NUM_HOUSES, 1.20, dtype=np.float32),
                np.full(NUM_HOUSES, 10.0, dtype=np.float32),
                np.full(NUM_DER, float(SOLAR_CAPACITY_KW), dtype=np.float32),
                np.array([1.0], dtype=np.float32),
            ]),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(NUM_DER,), dtype=np.float32)

        self.handles = build_grid()
        self.rng: np.random.Generator = np.random.default_rng(self.cfg.seed)

        self._hour: float = self.cfg.start_hour
        self._steps: int = 0
        self._max_steps: int = int(round(self.cfg.episode_hours / self.cfg.dt_hours))

        self._last_loads_kw: np.ndarray = np.zeros(NUM_HOUSES, dtype=np.float32)
        self._last_available_solar_kw: np.ndarray = np.zeros(NUM_DER, dtype=np.float32)
        self._last_voltages: np.ndarray = np.ones(NUM_HOUSES, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._hour = self.cfg.start_hour
        self._steps = 0

        # Sample the first timestep with *zero curtailment* just to populate obs.
        self._sample_exogenous()
        set_loads_kw(self.handles, self._last_loads_kw.tolist())
        set_solar_kw(self.handles, self._last_available_solar_kw.tolist())
        self._run_powerflow()

        return self._build_observation(), self._build_info(curtail=np.zeros(NUM_DER))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(NUM_DER)
        action = np.clip(action, 0.0, 1.0)

        # 1. Apply curtailment to current available solar
        curtailed_kw = self._last_available_solar_kw * action
        delivered_kw = self._last_available_solar_kw - curtailed_kw
        set_loads_kw(self.handles, self._last_loads_kw.tolist())
        set_solar_kw(self.handles, delivered_kw.tolist())

        # 2. Power flow
        converged = self._run_powerflow()

        # 3. Reward
        reward, violated = self._compute_reward(curtailed_kw, converged)

        # 4. Advance time + sample next exogenous so observation reflects the
        #    state the agent will see on the *next* call.
        self._steps += 1
        self._hour = (self._hour + self.cfg.dt_hours) % 24.0
        terminated = False
        truncated = self._steps >= self._max_steps

        self._sample_exogenous()
        obs = self._build_observation()
        info = self._build_info(
            curtail=action,
            curtailed_kw=curtailed_kw,
            delivered_kw=delivered_kw,
            violated=violated,
            converged=converged,
        )
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_exogenous(self) -> None:
        self._last_loads_kw = (
            load_profile(self._hour, rng=self.rng) * self.cfg.load_scale
        ).astype(np.float32)
        self._last_available_solar_kw = (
            solar_profile(self._hour, rng=self.rng) * self.cfg.solar_scale
        ).astype(np.float32)

    def _run_powerflow(self) -> bool:
        try:
            pp.runpp(self.handles.net, numba=False)
            v = self.handles.net.res_bus.vm_pu.loc[self.handles.house_buses].to_numpy()
            self._last_voltages = v.astype(np.float32)
            return True
        except Exception:
            # Diverged — keep last voltages, mark as violation in reward
            return False

    def _compute_reward(self, curtailed_kw: np.ndarray, converged: bool) -> tuple[float, bool]:
        w = self.reward_weights
        if not converged:
            return -100.0 * w["voltage_penalty_weight"], True

        v = self._last_voltages
        violations = int(np.sum((v < V_MIN) | (v > V_MAX)))
        violated = violations > 0

        reward = 0.0
        if violated:
            reward += -100.0 * w["voltage_penalty_weight"] * violations
        else:
            reward += w["stability_bonus"]
        reward += -w["curtailment_weight"] * float(np.sum(curtailed_kw))
        return reward, violated

    def _build_observation(self) -> np.ndarray:
        tod = np.array([self._hour / 24.0], dtype=np.float32)
        return np.concatenate([
            self._last_voltages,
            self._last_loads_kw,
            self._last_available_solar_kw,
            tod,
        ]).astype(np.float32)

    def _build_info(
        self,
        curtail: np.ndarray,
        curtailed_kw: np.ndarray | None = None,
        delivered_kw: np.ndarray | None = None,
        violated: bool = False,
        converged: bool = True,
    ) -> dict[str, Any]:
        return {
            "hour": float(self._hour),
            "step": int(self._steps),
            "voltages_pu": self._last_voltages.copy(),
            "loads_kw": self._last_loads_kw.copy(),
            "available_solar_kw": self._last_available_solar_kw.copy(),
            "curtail_action": curtail.copy(),
            "curtailed_kw": None if curtailed_kw is None else curtailed_kw.copy(),
            "delivered_kw": None if delivered_kw is None else delivered_kw.copy(),
            "violated": bool(violated),
            "converged": bool(converged),
        }

    # Convenience for verification / LLM layers later on.
    def set_reward_weights(self, **overrides: float) -> None:
        for k, v in overrides.items():
            if k not in DEFAULT_REWARD_WEIGHTS:
                raise KeyError(f"Unknown reward weight: {k}")
            self.reward_weights[k] = float(v)

    def set_scales(self, solar_scale: float | None = None, load_scale: float | None = None) -> None:
        if solar_scale is not None:
            self.cfg.solar_scale = float(solar_scale)
        if load_scale is not None:
            self.cfg.load_scale = float(load_scale)
