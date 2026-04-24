"""
Phase 1 — synthetic residential load and PV generation profiles.

All profiles are functions of a continuous `hour` in [0, 24). The environment
steps in `dt_hours` increments (default 1 h → 24 steps per episode).

Load pattern (0.5..4 kW):
    * night minimum
    * morning ramp (≈ 7 AM)
    * midday dip
    * evening peak (≈ 7 PM)

Solar pattern (0..5 kW):
    * zero outside [sunrise, sunset]
    * smooth sinusoidal bell shape
    * multiplicative cloud-noise term (stochastic)
"""
from __future__ import annotations

import numpy as np

from .network import NUM_DER, NUM_HOUSES, SOLAR_CAPACITY_KW


def load_profile(
    hour: float | np.ndarray,
    num_houses: int = NUM_HOUSES,
    rng: np.random.Generator | None = None,
    per_house_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Return a vector of length `num_houses` with P-load in **kW**."""
    h = np.asarray(hour, dtype=float)
    # Two-peak residential shape in kW
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
    return np.clip(loads, 0.5, 4.0)


def solar_profile(
    hour: float | np.ndarray,
    num_der: int = NUM_DER,
    rng: np.random.Generator | None = None,
    cloud_intensity: float = 0.15,
    capacity_kw: float = SOLAR_CAPACITY_KW,
) -> np.ndarray:
    """Return a vector of length `num_der` with available PV generation (kW).

    This is the *uncurtailed* upstream generation. The RL action later scales
    each value down via `(1 - curtailment[i])`.
    """
    h = float(hour)
    sunrise, sunset = 6.0, 19.0
    if h <= sunrise or h >= sunset:
        clear = 0.0
    else:
        x = (h - sunrise) / (sunset - sunrise)         # 0..1
        clear = capacity_kw * np.sin(np.pi * x) ** 1.3  # bell, peak ~= capacity

    if rng is None:
        rng = np.random.default_rng()
    clouds = rng.normal(loc=1.0, scale=cloud_intensity, size=num_der)
    clouds = np.clip(clouds, 0.2, 1.1)
    return np.clip(clear * clouds, 0.0, capacity_kw)


def sample_step(
    hour: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: one (loads_kw, solar_kw) sample for the given hour."""
    if rng is None:
        rng = np.random.default_rng()
    return load_profile(hour, rng=rng), solar_profile(hour, rng=rng)
