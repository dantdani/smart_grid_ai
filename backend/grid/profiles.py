"""
Phase 1 — synthetic residential-load and DER-availability profiles.

Profiles are functions of `hour` in [0, 24).  The environment steps in
`dt_hours` increments (default 1 h, 24 steps per episode).

Profile design
--------------
Load (per house, kW):
    night minimum, morning bump (~7 AM), midday dip, evening peak (~7 PM).

Solar (per panel, kW):
    zero outside [sunrise, sunset], smooth bell shape × cloud noise.
    -> automatically zero at night.

Wind (per turbine, kW):
    diurnal pattern peaking late afternoon / overnight (low pressure
    gradient typically picks up after sunset). Never identically zero.

Gas (per generator, kW):
    fully dispatchable. Profile reports nameplate capacity as "available";
    actual output is decided by the RL/OPF/operator action.
"""
from __future__ import annotations

import numpy as np

from .network import (
    DER_CAPACITY_KW,
    DER_HOUSES,
    DER_TYPES,
    NUM_DER,
    NUM_HOUSES,
    SOLAR_CAPACITY_KW,
)


# --------------------------------------------------------------------------- 
# Loads
# --------------------------------------------------------------------------- 
def load_profile(
    hour: float | np.ndarray,
    num_houses: int = NUM_HOUSES,
    rng: np.random.Generator | None = None,
    per_house_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Return a vector of length `num_houses` with P-load in **kW**."""
    h = np.asarray(hour, dtype=float)
    morning = 1.3 * np.exp(-0.5 * ((h - 7.5) / 1.2) ** 2)
    evening = 2.0 * np.exp(-0.5 * ((h - 19.0) / 1.5) ** 2)
    base = 0.6
    profile_kw = base + morning + evening  # ~0.6 .. ~3.2 kW

    if per_house_scale is None:
        per_house_scale = np.ones(num_houses)

    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(loc=1.0, scale=0.08, size=num_houses)
    loads = profile_kw * per_house_scale * noise
    # Allow generous headroom so per-house scales can drive a demand spike.
    return np.clip(loads, 0.0, 20.0)


# --------------------------------------------------------------------------- 
# Per-tech DER profiles
# --------------------------------------------------------------------------- 
def solar_kw(hour: float, rng: np.random.Generator, capacity_kw: float) -> float:
    sunrise, sunset = 6.0, 19.0
    if hour <= sunrise or hour >= sunset:
        return 0.0
    x = (hour - sunrise) / (sunset - sunrise)
    clear = capacity_kw * float(np.sin(np.pi * x) ** 1.3)
    cloud = float(np.clip(rng.normal(loc=1.0, scale=0.15), 0.2, 1.1))
    return float(np.clip(clear * cloud, 0.0, capacity_kw))


def wind_kw(hour: float, rng: np.random.Generator, capacity_kw: float) -> float:
    """Wind: bimodal diurnal cycle (afternoon + overnight), with gust noise."""
    # Two soft Gaussian bumps at hour 3 (overnight) and hour 16 (afternoon)
    overnight = 0.6 * np.exp(-0.5 * ((hour - 3.0) / 4.0) ** 2)
    afternoon = 0.55 * np.exp(-0.5 * ((hour - 16.0) / 3.5) ** 2)
    base = 0.25                                 # always at least a breeze
    fraction = base + overnight + afternoon     # ~0.25 .. ~0.95
    gust = float(np.clip(rng.normal(loc=1.0, scale=0.18), 0.3, 1.25))
    return float(np.clip(fraction * gust * capacity_kw, 0.0, capacity_kw))


def gas_kw(hour: float, rng: np.random.Generator, capacity_kw: float) -> float:
    """Natural-gas micro-generator: nameplate is always available."""
    return float(capacity_kw)


_DER_FN = {"solar": solar_kw, "wind": wind_kw, "gas": gas_kw}


def der_availability(
    hour: float,
    rng: np.random.Generator | None = None,
    der_types: list[str] = DER_TYPES,
    capacity_kw: list[float] = DER_CAPACITY_KW,
) -> np.ndarray:
    """Return per-DER available kW (length = NUM_DER) at the given hour."""
    if rng is None:
        rng = np.random.default_rng()
    out = np.empty(len(der_types), dtype=float)
    for i, (kind, cap) in enumerate(zip(der_types, capacity_kw)):
        out[i] = _DER_FN[kind](float(hour), rng, float(cap))
    return out


# --------------------------------------------------------------------------- 
# Backward-compatible solar-only profile (still used by older smoke scripts)
# --------------------------------------------------------------------------- 
def solar_profile(
    hour: float | np.ndarray,
    num_der: int = NUM_DER,
    rng: np.random.Generator | None = None,
    cloud_intensity: float = 0.15,
    capacity_kw: float = SOLAR_CAPACITY_KW,
) -> np.ndarray:
    """Legacy: pretend every DER is solar. New code should use `der_availability`."""
    if rng is None:
        rng = np.random.default_rng()
    return np.array(
        [solar_kw(float(hour), rng, float(capacity_kw)) for _ in range(num_der)],
        dtype=float,
    )


def sample_step(
    hour: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: one (loads_kw, der_kw) sample for the given hour."""
    if rng is None:
        rng = np.random.default_rng()
    return load_profile(hour, rng=rng), der_availability(hour, rng=rng)
