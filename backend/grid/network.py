"""
Phase 1a — pandapower radial distribution feeder.

Topology
--------
* 1 slack bus (main grid, 20 kV MV side) -> MV/LV transformer -> LV feeder (0.4 kV)
* 10 LV load buses, one per house, connected in a radial chain
* Residential loads on every house
* Solar sgens (0..5 kW) on the 5 DER houses only

House indexing is 1..10 (as in the spec). `DER_HOUSES = [1, 3, 5, 7, 9]`.

All helper setters (`set_loads_kw`, `set_solar_kw`) accept per-house lists in
**kW** and convert to pandapower's MW internally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandapower as pp


NUM_HOUSES: int = 10
DER_HOUSES: list[int] = [1, 3, 5, 7, 9]  # 1-indexed house numbers with solar
NUM_DER: int = len(DER_HOUSES)
SOLAR_CAPACITY_KW: float = 5.0  # per-panel nameplate, 0..5 kW

# Feeder electrical parameters — chosen so that ~4 kW reverse flow per DER
# house produces a visible voltage rise that can cross 1.05 p.u.
_MV_KV: float = 20.0
_LV_KV: float = 0.4
# LV feeder tuned so that (a) evening residential load alone stays inside
# [0.95, 1.05] p.u., and (b) full uncurtailed midday PV on the 5 DER houses
# pushes the tail-end bus above 1.05 p.u.  The slack bus is held at 1.02 p.u.
# (typical LV setpoint) so the normal operating band sits comfortably inside
# the safe window and only reverse-flow causes violations.
_LINE_LEN_KM: float = 0.08        # 80 m between adjacent houses
_R_OHM_PER_KM: float = 0.9        # moderately thin LV cable
_X_OHM_PER_KM: float = 0.09
_MAX_I_KA: float = 0.2
_SLACK_VM_PU: float = 1.025


@dataclass
class GridHandles:
    """Handles into the pandapower net so the env can update it fast."""

    net: pp.pandapowerNet
    slack_bus: int
    house_buses: list[int] = field(default_factory=list)   # len = NUM_HOUSES
    load_indices: list[int] = field(default_factory=list)  # len = NUM_HOUSES
    sgen_indices: list[int] = field(default_factory=list)  # len = NUM_DER


def build_grid() -> GridHandles:
    """Create the pandapower net and return handles to its mutable elements."""
    net = pp.create_empty_network(name="smart_grid_ai_feeder")

    # --- Buses -------------------------------------------------------------
    mv_bus = pp.create_bus(net, vn_kv=_MV_KV, name="MV_Slack")
    lv_root = pp.create_bus(net, vn_kv=_LV_KV, name="LV_Root")

    house_buses: list[int] = []
    for h in range(1, NUM_HOUSES + 1):
        house_buses.append(pp.create_bus(net, vn_kv=_LV_KV, name=f"House_{h}"))

    # --- Slack -------------------------------------------------------------
    pp.create_ext_grid(net, bus=mv_bus, vm_pu=_SLACK_VM_PU, name="MainGrid")

    # --- MV/LV transformer -------------------------------------------------
    pp.create_transformer_from_parameters(
        net,
        hv_bus=mv_bus,
        lv_bus=lv_root,
        sn_mva=0.25,        # 250 kVA distribution transformer
        vn_hv_kv=_MV_KV,
        vn_lv_kv=_LV_KV,
        vkr_percent=1.0,
        vk_percent=4.0,
        pfe_kw=0.5,
        i0_percent=0.1,
        name="MV_LV_Trafo",
    )

    # --- Radial LV feeder: LV_Root -> House_1 -> House_2 -> ... -> House_10
    prev_bus = lv_root
    for idx, bus in enumerate(house_buses, start=1):
        pp.create_line_from_parameters(
            net,
            from_bus=prev_bus,
            to_bus=bus,
            length_km=_LINE_LEN_KM,
            r_ohm_per_km=_R_OHM_PER_KM,
            x_ohm_per_km=_X_OHM_PER_KM,
            c_nf_per_km=0.0,
            max_i_ka=_MAX_I_KA,
            name=f"Line_{idx}",
        )
        prev_bus = bus

    # --- Residential loads (placeholders, updated every step) -------------
    load_indices: list[int] = []
    for h, bus in enumerate(house_buses, start=1):
        load_indices.append(
            pp.create_load(
                net,
                bus=bus,
                p_mw=1e-3,        # 1 kW default, overwritten per step
                q_mvar=0.3e-3,    # cos(phi) ~ 0.95
                name=f"Load_H{h}",
            )
        )

    # --- Solar sgens on DER houses only -----------------------------------
    sgen_indices: list[int] = []
    for h in DER_HOUSES:
        bus = house_buses[h - 1]
        sgen_indices.append(
            pp.create_sgen(
                net,
                bus=bus,
                p_mw=0.0,
                q_mvar=0.0,
                name=f"PV_H{h}",
                type="PV",
                max_p_mw=SOLAR_CAPACITY_KW * 1e-3,
                min_p_mw=0.0,
                controllable=True,   # required for runopp
            )
        )

    # Sanity-check power flow once at build time.
    pp.runpp(net)

    return GridHandles(
        net=net,
        slack_bus=mv_bus,
        house_buses=house_buses,
        load_indices=load_indices,
        sgen_indices=sgen_indices,
    )


def set_loads_kw(handles: GridHandles, loads_kw: Sequence[float]) -> None:
    """Set residential P-load for every house (kW). Q scales with cos(phi)=0.95."""
    if len(loads_kw) != NUM_HOUSES:
        raise ValueError(f"Expected {NUM_HOUSES} load values, got {len(loads_kw)}.")
    loads = np.asarray(loads_kw, dtype=float)
    q_frac = np.tan(np.arccos(0.95))  # Q/P at cos(phi)=0.95
    handles.net.load.loc[handles.load_indices, "p_mw"] = loads * 1e-3
    handles.net.load.loc[handles.load_indices, "q_mvar"] = loads * 1e-3 * q_frac


def set_solar_kw(handles: GridHandles, solar_kw: Sequence[float]) -> None:
    """Set solar P-injection for each DER house (kW)."""
    if len(solar_kw) != NUM_DER:
        raise ValueError(f"Expected {NUM_DER} solar values, got {len(solar_kw)}.")
    solar = np.clip(np.asarray(solar_kw, dtype=float), 0.0, SOLAR_CAPACITY_KW)
    handles.net.sgen.loc[handles.sgen_indices, "p_mw"] = solar * 1e-3
    handles.net.sgen.loc[handles.sgen_indices, "q_mvar"] = 0.0
