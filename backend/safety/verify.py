"""
Phase 3 — Physics verification + OPF safety override.

Flow:
    proposed curtailment (from RL)
        -> simulate with pandapower.runpp()
        -> if all voltages in [V_MIN, V_MAX]:   APPLY RL
        -> else:                                run OPF, derive safe curtailment, APPLY OPF
        -> if OPF fails too:                    fall back to full curtailment

The verifier *does not* mutate the environment. It clones the pandapower net,
probes it, and returns the action that should actually be applied along with a
structured decision record for the UI / log.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandapower as pp

from ..grid import DER_HOUSES, NUM_HOUSES, SOLAR_CAPACITY_KW, set_loads_kw, set_solar_kw
from ..rl.env import V_MAX, V_MIN, SmartGridEnv

NUM_DER = len(DER_HOUSES)

Source = Literal["rl", "opf", "fallback"]


@dataclass
class VerificationResult:
    source: Source                    # which layer produced the applied action
    applied_curtailment: list[float]  # 5 floats in [0, 1]
    rl_curtailment: list[float]
    predicted_voltages: list[float]   # per-house voltages under the APPLIED action
    rl_voltages: list[float] | None   # per-house voltages under the RL action
    rl_violated: bool
    opf_used: bool
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def _simulate(net: pp.pandapowerNet, handles, loads_kw, available_kw, curtail) -> tuple[bool, np.ndarray]:
    """Apply (loads, solar * (1-curtail)) to *net* and run power flow. Returns (converged, house_voltages)."""
    delivered = np.asarray(available_kw, dtype=float) * (1.0 - np.asarray(curtail, dtype=float))
    set_loads_kw(handles, loads_kw)
    set_solar_kw(handles, delivered.tolist())
    try:
        pp.runpp(net, numba=False)
    except Exception:
        return False, np.full(NUM_HOUSES, np.nan)
    v = net.res_bus.vm_pu.loc[handles.house_buses].to_numpy()
    return True, v


def _any_violation(v: np.ndarray) -> bool:
    if np.any(np.isnan(v)):
        return True
    return bool(np.any((v < V_MIN) | (v > V_MAX)))


def _clone_handles(env: SmartGridEnv):
    """Deep-copy the env's net + handle indices so verification doesn't touch live state."""
    clone = copy.deepcopy(env.handles)
    return clone


def _run_opf(handles, loads_kw, available_kw) -> np.ndarray | None:
    """
    Use pandapower OPF to find the minimum-curtailment PV dispatch that keeps
    all bus voltages in [V_MIN, V_MAX]. Returns per-DER curtailment in [0,1],
    or None if OPF fails.

    Model:
      - Cap each sgen's p_mw at its currently available PV (upper bound).
      - Minimize cost = -p_mw (i.e. maximize delivered PV) on each sgen.
      - Bus voltage box: [V_MIN, V_MAX].
    """
    net = handles.net
    avail = np.clip(np.asarray(available_kw, dtype=float), 0.0, SOLAR_CAPACITY_KW)

    set_loads_kw(handles, loads_kw)
    # Upper-bound sgen injection at the currently available PV
    for i, sidx in enumerate(handles.sgen_indices):
        net.sgen.at[sidx, "max_p_mw"] = float(avail[i]) * 1e-3
        net.sgen.at[sidx, "min_p_mw"] = 0.0
        net.sgen.at[sidx, "controllable"] = True
        net.sgen.at[sidx, "p_mw"] = float(avail[i]) * 1e-3

    # Voltage box constraints on every bus
    net.bus["min_vm_pu"] = V_MIN
    net.bus["max_vm_pu"] = V_MAX

    # Objective: maximize total sgen P (== minimize curtailment).
    # pandapower OPF minimizes sum(cp1_eur_per_mw * p_mw), so set cp1 = -1 on sgens.
    if hasattr(net, "poly_cost"):
        net.poly_cost = net.poly_cost.iloc[0:0]   # clear any previous
    for sidx in handles.sgen_indices:
        pp.create_poly_cost(net, element=sidx, et="sgen",
                            cp1_eur_per_mw=-1.0, cp0_eur=0.0)

    # Allow slack to adjust freely (no cost) so the OPF has a feasible solution.
    for ext_idx in net.ext_grid.index:
        pp.create_poly_cost(net, element=ext_idx, et="ext_grid",
                            cp1_eur_per_mw=0.0, cp0_eur=0.0)

    try:
        pp.runopp(net, verbose=False, numba=False)
    except Exception:
        return None

    dispatched_mw = net.res_sgen.p_mw.loc[handles.sgen_indices].to_numpy()
    delivered_kw = np.clip(dispatched_mw * 1e3, 0.0, avail)
    # curtailment = 1 - delivered/available  (0 if available is 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        curtail = np.where(avail > 1e-9, 1.0 - delivered_kw / avail, 0.0)
    return np.clip(curtail, 0.0, 1.0)


def verify_action(
    env: SmartGridEnv,
    rl_action: np.ndarray,
) -> VerificationResult:
    """Return the action that should actually be applied + decision metadata."""
    rl = np.clip(np.asarray(rl_action, dtype=float), 0.0, 1.0)
    loads_kw = env._last_loads_kw.tolist()
    available_kw = env._last_available_solar_kw.tolist()

    # 1. Probe the RL action on a cloned net
    probe = _clone_handles(env)
    ok_rl, v_rl = _simulate(probe.net, probe, loads_kw, available_kw, rl)
    rl_violated = _any_violation(v_rl) if ok_rl else True

    if ok_rl and not rl_violated:
        return VerificationResult(
            source="rl",
            applied_curtailment=rl.tolist(),
            rl_curtailment=rl.tolist(),
            predicted_voltages=v_rl.tolist(),
            rl_voltages=v_rl.tolist(),
            rl_violated=False,
            opf_used=False,
            message="RL action verified safe.",
        )

    # 2. RL unsafe -> run OPF on a fresh clone
    opf_clone = _clone_handles(env)
    opf_curtail = _run_opf(opf_clone, loads_kw, available_kw)

    if opf_curtail is not None:
        check = _clone_handles(env)
        ok_opf, v_opf = _simulate(check.net, check, loads_kw, available_kw, opf_curtail)
        if ok_opf and not _any_violation(v_opf):
            return VerificationResult(
                source="opf",
                applied_curtailment=opf_curtail.tolist(),
                rl_curtailment=rl.tolist(),
                predicted_voltages=v_opf.tolist(),
                rl_voltages=(v_rl.tolist() if ok_rl else None),
                rl_violated=True,
                opf_used=True,
                message="RL action unsafe; OPF override applied.",
            )

    # 3. Last-resort fallback: full curtailment
    fb = np.ones(NUM_DER)
    check = _clone_handles(env)
    _, v_fb = _simulate(check.net, check, loads_kw, available_kw, fb)
    return VerificationResult(
        source="fallback",
        applied_curtailment=fb.tolist(),
        rl_curtailment=rl.tolist(),
        predicted_voltages=v_fb.tolist(),
        rl_voltages=(v_rl.tolist() if ok_rl else None),
        rl_violated=True,
        opf_used=True,
        message="RL unsafe and OPF infeasible; full-curtailment fallback applied.",
    )
