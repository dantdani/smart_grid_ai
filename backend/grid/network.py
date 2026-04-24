"""
Phase 1a — pandapower radial distribution feeder.

Topology
--------
* 1 slack bus (main grid, 20 kV MV side) -> MV/LV transformer -> LV feeder (0.4 kV)
* 10 LV load buses, one per house, connected in a radial chain
* Residential loads on every house
* 5 DERs (one per "DER house"), in a mixed portfolio:
    - 3 rooftop solar (PV)
    - 1 small wind turbine
    - 1 natural-gas micro-generator

House indexing is 1..10 (as in the spec). `DER_HOUSES = [1, 3, 5, 7, 9]`,
`DER_TYPES = ["solar", "solar", "solar", "wind", "gas"]`.

All helper setters accept per-house lists in **kW** and convert to
pandapower's MW internally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandapower as pp


NUM_HOUSES: int = 10
DER_HOUSES: list[int] = [1, 3, 5, 7, 9]  # 1-indexed house numbers with a DER
DER_TYPES: list[str] = ["solar", "solar", "solar", "wind", "gas"]
NUM_DER: int = len(DER_HOUSES)

# Per-DER nameplate capacity (kW). Solar/wind sized so a sunny/windy day can
# push the feeder above 1.05 p.u. without curtailment; gas sized for night
# backup so an operator can demo "fire up the gas" during a demand spike.
DER_CAPACITY_KW: list[float] = [5.0, 5.0, 5.0, 5.0, 10.0]
SOLAR_CAPACITY_KW: float = 5.0  # kept for backward-compat re-exports

# Feeder electrical parameters (unchanged from earlier phases) ---------------
_MV_KV: float = 20.0
_LV_KV: float = 0.4
_LINE_LEN_KM: float = 0.08
_R_OHM_PER_KM: float = 0.9
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
    der_types: list[str] = field(default_factory=list)     # len = NUM_DER
    der_capacity_kw: list[float] = field(default_factory=list)  # len = NUM_DER


def build_grid() -> GridHandles:
    """Create the pandapower net and return handles to its mutable elements."""
    net = pp.create_empty_network(name="smart_grid_ai_feeder")

    mv_bus = pp.create_bus(net, vn_kv=_MV_KV, name="MV_Slack")
    lv_root = pp.create_bus(net, vn_kv=_LV_KV, name="LV_Root")

    house_buses: list[int] = []
    for h in range(1, NUM_HOUSES + 1):
        house_buses.append(pp.create_bus(net, vn_kv=_LV_KV, name=f"House_{h}"))

    pp.create_ext_grid(net, bus=mv_bus, vm_pu=_SLACK_VM_PU, name="MainGrid")

    pp.create_transformer_from_parameters(
        net,
        hv_bus=mv_bus, lv_bus=lv_root,
        sn_mva=0.25, vn_hv_kv=_MV_KV, vn_lv_kv=_LV_KV,
        vkr_percent=1.0, vk_percent=4.0, pfe_kw=0.5, i0_percent=0.1,
        name="MV_LV_Trafo",
    )

    prev_bus = lv_root
    for idx, bus in enumerate(house_buses, start=1):
        pp.create_line_from_parameters(
            net,
            from_bus=prev_bus, to_bus=bus,
            length_km=_LINE_LEN_KM,
            r_ohm_per_km=_R_OHM_PER_KM,
            x_ohm_per_km=_X_OHM_PER_KM,
            c_nf_per_km=0.0,
            max_i_ka=_MAX_I_KA,
            name=f"Line_{idx}",
        )
        prev_bus = bus

    load_indices: list[int] = []
    for h, bus in enumerate(house_buses, start=1):
        load_indices.append(
            pp.create_load(
                net, bus=bus,
                p_mw=1e-3, q_mvar=0.3e-3,
                name=f"Load_H{h}",
            )
        )

    # All DERs modelled as controllable sgens. `type` is set per technology so
    # the API/UI can label them; OPF treats them as generic dispatchable.
    sgen_indices: list[int] = []
    pp_type_map = {"solar": "PV", "wind": "WP", "gas": "Gas"}
    for h, kind, cap in zip(DER_HOUSES, DER_TYPES, DER_CAPACITY_KW):
        bus = house_buses[h - 1]
        sgen_indices.append(
            pp.create_sgen(
                net, bus=bus,
                p_mw=0.0, q_mvar=0.0,
                name=f"{kind.upper()}_H{h}",
                type=pp_type_map[kind],
                max_p_mw=float(cap) * 1e-3,
                min_p_mw=0.0,
                controllable=True,
            )
        )

    pp.runpp(net)

    return GridHandles(
        net=net,
        slack_bus=mv_bus,
        house_buses=house_buses,
        load_indices=load_indices,
        sgen_indices=sgen_indices,
        der_types=list(DER_TYPES),
        der_capacity_kw=list(DER_CAPACITY_KW),
    )


def set_loads_kw(handles: GridHandles, loads_kw: Sequence[float]) -> None:
    """Set residential P-load for every house (kW). Q scales with cos(phi)=0.95."""
    if len(loads_kw) != NUM_HOUSES:
        raise ValueError(f"Expected {NUM_HOUSES} load values, got {len(loads_kw)}.")
    loads = np.asarray(loads_kw, dtype=float)
    q_frac = np.tan(np.arccos(0.95))
    handles.net.load.loc[handles.load_indices, "p_mw"] = loads * 1e-3
    handles.net.load.loc[handles.load_indices, "q_mvar"] = loads * 1e-3 * q_frac


def set_der_kw(handles: GridHandles, der_kw: Sequence[float]) -> None:
    """Set each DER's P-injection (kW), clipped to its nameplate capacity."""
    if len(der_kw) != NUM_DER:
        raise ValueError(f"Expected {NUM_DER} DER values, got {len(der_kw)}.")
    caps = np.asarray(handles.der_capacity_kw, dtype=float)
    p = np.clip(np.asarray(der_kw, dtype=float), 0.0, caps)
    handles.net.sgen.loc[handles.sgen_indices, "p_mw"] = p * 1e-3
    handles.net.sgen.loc[handles.sgen_indices, "q_mvar"] = 0.0


# Backward-compatible alias used by older code paths / tests.
def set_solar_kw(handles: GridHandles, solar_kw: Sequence[float]) -> None:
    """Alias of :func:`set_der_kw` (legacy name)."""
    set_der_kw(handles, solar_kw)
