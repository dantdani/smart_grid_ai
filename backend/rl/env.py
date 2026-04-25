"""
Phase 1b — Gymnasium wrapper around the pandapower feeder.

Observation (26,) = [
    10 bus voltages (p.u.),
    10 household loads (kW),
     5 available DER kW (mixed: 3 solar + 1 wind + 1 gas, pre-curtailment),
     1 time_of_day in [0, 1),
]

Action (5,) in [0, 1] — per-DER curtailment fraction.
    0.0 => no curtailment (inject full available power)
    1.0 => full curtailment (inject 0 kW)

The action shape is preserved across all DER technologies, so a policy
trained on the pure-solar version still loads cleanly. The user can also
*pin* one or more DERs (`manual_overrides`) so the RL agent only optimises
the remaining ones.

Reward
------
* -100 * voltage_penalty_weight per bus voltage outside [V_MIN, V_MAX]
* -curtailment_weight * total_curtailed_kW
* +stability_bonus when every bus is in the safe band
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandapower as pp
from gymnasium import spaces

from ..grid import (
    DER_CAPACITY_KW,
    DER_HOUSES,
    DER_TYPES,
    NUM_DER,
    NUM_HOUSES,
    SOLAR_CAPACITY_KW,
    build_grid,
    der_availability,
    load_profile,
    set_der_kw,
    set_loads_kw,
)

V_MIN: float = 0.95
V_MAX: float = 1.05
OBS_DIM: int = NUM_HOUSES + NUM_HOUSES + NUM_DER + 1  # 26

DEFAULT_REWARD_WEIGHTS = {
    "voltage_penalty_weight": 1.0,
    "curtailment_weight": 0.1,
    "stability_bonus": 10.0,
}


@dataclass
class EnvConfig:
    dt_hours: float = 1.0
    episode_hours: float = 24.0
    start_hour: float = 0.0
    solar_scale: float = 1.0      # multiplies solar DER availability only
    wind_scale: float = 1.0       # multiplies wind DER availability only
    gas_scale: float = 1.0        # multiplies gas DER availability only (dispatchable cap)
    load_scale: float = 1.0       # global load multiplier (per-house overrides on top)
    seed: int | None = None


class SmartGridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.reward_weights = dict(DEFAULT_REWARD_WEIGHTS)

        max_caps = np.asarray(DER_CAPACITY_KW, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([
                np.full(NUM_HOUSES, 0.80, dtype=np.float32),
                np.full(NUM_HOUSES, 0.0, dtype=np.float32),
                np.zeros(NUM_DER, dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            ]),
            high=np.concatenate([
                np.full(NUM_HOUSES, 1.20, dtype=np.float32),
                np.full(NUM_HOUSES, 30.0, dtype=np.float32),
                max_caps,
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
        self._last_available_der_kw: np.ndarray = np.zeros(NUM_DER, dtype=np.float32)
        self._last_voltages: np.ndarray = np.ones(NUM_HOUSES, dtype=np.float32)

        # Per-house multiplier (defaults to 1.0 each). Operator slider sets these.
        self.house_load_scales: np.ndarray = np.ones(NUM_HOUSES, dtype=np.float32)
        # NaN means "RL controls it"; finite value pins delivered power in kW for that DER.
        # Pinned DERs bypass availability (so an operator can demand fixed gas/wind output
        # even at night, or hold solar to a fixed kW even when availability would be higher).
        self.manual_overrides: np.ndarray = np.full(NUM_DER, np.nan, dtype=np.float32)

    # Backward-compat alias (older code may still read this attribute name)
    @property
    def _last_available_solar_kw(self) -> np.ndarray:  # pragma: no cover - shim
        return self._last_available_der_kw

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

        self._sample_exogenous()
        set_loads_kw(self.handles, self._last_loads_kw.tolist())
        set_der_kw(self.handles, self._last_available_der_kw.tolist())
        self._run_powerflow()

        return self._build_observation(), self._build_info(curtail=np.zeros(NUM_DER))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(NUM_DER)
        action = np.clip(action, 0.0, 1.0)

        # Start with RL-derived dispatch from availability + curtailment.
        delivered_kw = self._last_available_der_kw * (1.0 - action)

        # Pinned DERs override delivered power outright (bypass availability + RL).
        caps = np.asarray(DER_CAPACITY_KW, dtype=np.float32)
        mask = ~np.isnan(self.manual_overrides)
        if mask.any():
            pinned_kw = np.clip(self.manual_overrides[mask], 0.0, caps[mask])
            delivered_kw[mask] = pinned_kw

        # Recompute action so logs/observations reflect what was actually applied.
        # For pinned DERs we synthesize an effective curtailment fraction relative to
        # the nameplate capacity (so the action vector still sits in [0,1]).
        with np.errstate(divide="ignore", invalid="ignore"):
            applied_action = np.where(
                self._last_available_der_kw > 1e-9,
                1.0 - delivered_kw / self._last_available_der_kw,
                np.where(caps > 1e-9, 1.0 - delivered_kw / caps, 0.0),
            )
        applied_action = np.clip(applied_action, 0.0, 1.0).astype(np.float32)

        # "Curtailed" power for the reward is what we *could* have produced minus what we did.
        # For a pinned DER that exceeds availability (e.g. gas at night), curtailed = 0.
        effective_avail = np.maximum(self._last_available_der_kw, delivered_kw)
        curtailed_kw = np.maximum(effective_avail - delivered_kw, 0.0)

        set_loads_kw(self.handles, self._last_loads_kw.tolist())
        set_der_kw(self.handles, delivered_kw.tolist())

        converged = self._run_powerflow()
        reward, violated = self._compute_reward(curtailed_kw, converged)

        self._steps += 1
        self._hour = (self._hour + self.cfg.dt_hours) % 24.0
        terminated = False
        truncated = self._steps >= self._max_steps

        self._sample_exogenous()
        obs = self._build_observation()
        info = self._build_info(
            curtail=applied_action,
            curtailed_kw=curtailed_kw,
            delivered_kw=delivered_kw,
            violated=violated,
            converged=converged,
        )
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Overrides / scales
    # ------------------------------------------------------------------
    def apply_overrides(self, action: np.ndarray) -> np.ndarray:
        """Substitute pinned DERs' equivalent curtailment into the RL action.

        Used by the safety verifier so its RL-probe simulation reflects what
        will actually be dispatched. Pinned DERs are converted from kW to a
        curtailment fraction relative to current availability (or nameplate
        when availability is zero, which is the case e.g. for gas at night).
        """
        out = np.asarray(action, dtype=np.float32).copy()
        mask = ~np.isnan(self.manual_overrides)
        if not mask.any():
            return out
        caps = np.asarray(DER_CAPACITY_KW, dtype=np.float32)
        avail = self._last_available_der_kw
        pinned_kw = np.clip(self.manual_overrides[mask], 0.0, caps[mask])
        denom = np.where(avail[mask] > 1e-9, avail[mask], caps[mask])
        with np.errstate(divide="ignore", invalid="ignore"):
            curt = np.where(denom > 1e-9, 1.0 - pinned_kw / denom, 0.0)
        out[mask] = np.clip(curt, 0.0, 1.0)
        return out

    def set_house_load_scale(self, house_id: int, scale: float) -> None:
        if not (1 <= house_id <= NUM_HOUSES):
            raise ValueError(f"house_id must be 1..{NUM_HOUSES}, got {house_id}")
        self.house_load_scales[house_id - 1] = float(np.clip(scale, 0.0, 5.0))

    def set_house_load_scales(self, scales) -> None:
        arr = np.asarray(scales, dtype=np.float32).reshape(-1)
        if arr.size != NUM_HOUSES:
            raise ValueError(f"Expected {NUM_HOUSES} scales, got {arr.size}")
        self.house_load_scales = np.clip(arr, 0.0, 5.0)

    def set_der_override(self, der_index: int, power_kw: float | None) -> None:
        """Pin a DER to a fixed delivered power in kW (None clears the pin)."""
        if not (0 <= der_index < NUM_DER):
            raise ValueError(f"der_index must be 0..{NUM_DER - 1}, got {der_index}")
        if power_kw is None:
            self.manual_overrides[der_index] = np.nan
        else:
            cap = float(DER_CAPACITY_KW[der_index])
            self.manual_overrides[der_index] = float(np.clip(power_kw, 0.0, cap))

    def clear_der_overrides(self) -> None:
        self.manual_overrides[:] = np.nan

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_exogenous(self) -> None:
        loads = load_profile(
            self._hour,
            rng=self.rng,
            per_house_scale=self.house_load_scales.astype(float),
        ) * self.cfg.load_scale
        self._last_loads_kw = loads.astype(np.float32)

        avail = der_availability(self._hour, rng=self.rng)
        # Per-tech multipliers (solar_scale only applies to solar DERs so
        # "no solar at night" stays true; same idea for wind/gas).
        solar_mask = np.array([t == "solar" for t in DER_TYPES])
        wind_mask = np.array([t == "wind" for t in DER_TYPES])
        gas_mask = np.array([t == "gas" for t in DER_TYPES])
        avail[solar_mask] *= self.cfg.solar_scale
        avail[wind_mask] *= self.cfg.wind_scale
        avail[gas_mask] *= self.cfg.gas_scale
        self._last_available_der_kw = avail.astype(np.float32)

    def _run_powerflow(self) -> bool:
        try:
            pp.runpp(self.handles.net, numba=False)
            v = self.handles.net.res_bus.vm_pu.loc[self.handles.house_buses].to_numpy()
            self._last_voltages = v.astype(np.float32)
            return True
        except Exception:
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
            self._last_available_der_kw,
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
            "available_solar_kw": self._last_available_der_kw.copy(),  # legacy key
            "available_der_kw": self._last_available_der_kw.copy(),
            "der_types": list(DER_TYPES),
            "curtail_action": curtail.copy(),
            "curtailed_kw": None if curtailed_kw is None else curtailed_kw.copy(),
            "delivered_kw": None if delivered_kw is None else delivered_kw.copy(),
            "violated": bool(violated),
            "converged": bool(converged),
        }

    def set_reward_weights(self, **overrides: float) -> None:
        for k, v in overrides.items():
            if k not in DEFAULT_REWARD_WEIGHTS:
                raise KeyError(f"Unknown reward weight: {k}")
            self.reward_weights[k] = float(v)

    def set_scales(
        self,
        solar_scale: float | None = None,
        load_scale: float | None = None,
        wind_scale: float | None = None,
        gas_scale: float | None = None,
    ) -> None:
        if solar_scale is not None:
            self.cfg.solar_scale = float(solar_scale)
        if load_scale is not None:
            self.cfg.load_scale = float(load_scale)
        if wind_scale is not None:
            self.cfg.wind_scale = float(wind_scale)
        if gas_scale is not None:
            self.cfg.gas_scale = float(gas_scale)
