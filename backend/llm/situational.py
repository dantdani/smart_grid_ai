"""
LLM Situational Awareness Module.

Converts a natural-language description of an environmental event
(storm, heatwave, fuel disruption, ...) into a structured JSON object
that can be applied to the SmartGridEnv simulation.

This module does NOT control the grid directly. It only produces
environment parameter updates; the RL controller reacts to the new
conditions.

Output schema (any subset; missing keys mean "no change"):
    {
        "event_type": str,                       # see EVENT_TYPES
        "solar_scale": float in [0.0, 1.0],
        "wind_scale":  float in [0.0, 1.5],
        "gas_scale":   float in [0.5, 1.5],
        "load_multiplier": float in [0.5, 2.0],
        "voltage_penalty_weight": float in [1.0, 5.0],
        "curtailment_penalty_weight": float in [0.1, 2.0]
    }

If the event has no grid impact, returns:
    {"event_type": "none", "status": "no_environment_change"}
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

EVENT_TYPES = (
    "storm",
    "cloud_cover",
    "clear_sky",
    "heatwave",
    "cold_wave",
    "high_wind",
    "low_wind",
    "grid_emergency",
    "fuel_disruption",
    "equipment_failure",
    "none",
)

# (key, lo, hi)
_RANGES: dict[str, tuple[float, float]] = {
    "solar_scale": (0.0, 1.0),
    "wind_scale": (0.0, 1.5),
    "gas_scale": (0.5, 1.5),
    "load_multiplier": (0.5, 2.0),
    "voltage_penalty_weight": (1.0, 5.0),
    "curtailment_penalty_weight": (0.1, 2.0),
}

_SYSTEM_PROMPT = """You are the Situational Awareness layer of a smart-grid simulator.

The grid has 10 houses and 5 DERs:
  DER 1 = natural gas generator (dispatchable, weather-independent)
  DER 2 = wind turbine
  DER 3, 4, 5 = solar panels

You receive a natural-language description of an environmental event and must
predict its impact on grid generation and demand. Output ONLY a JSON object
(no prose, no explanations) with a subset of these keys:

  event_type                  one of: storm, cloud_cover, clear_sky, heatwave,
                              cold_wave, high_wind, low_wind, grid_emergency,
                              fuel_disruption, equipment_failure, none
  solar_scale                 float in [0.0, 1.0]   (1.0 = full sun)
  wind_scale                  float in [0.0, 1.5]   (1.0 = nominal wind)
  gas_scale                   float in [0.5, 1.5]   (1.0 = full fuel availability)
  load_multiplier             float in [0.5, 2.0]   (1.0 = nominal demand)
  voltage_penalty_weight      float in [1.0, 5.0]   (higher = stricter voltage safety)
  curtailment_penalty_weight  float in [0.1, 2.0]   (higher = avoid curtailing renewables)

Rules:
  - clouds / storms / night          -> lower solar_scale
  - clear sky / sunny                 -> raise solar_scale
  - storms / strong wind              -> raise wind_scale
  - calm / low wind                   -> lower wind_scale
  - heatwave (AC) / cold wave (heat)  -> raise load_multiplier
  - fuel disruption / pipeline issue  -> lower gas_scale
  - storm / grid emergency            -> raise voltage_penalty_weight
  - equipment failure                 -> raise voltage_penalty_weight, lower
                                         affected DER scale
  - if nothing on the grid is affected, return
        {"event_type": "none", "status": "no_environment_change"}

Examples:

input: "Heavy storm approaching with strong winds and dense cloud cover."
output: {"event_type":"storm","solar_scale":0.3,"wind_scale":1.2,"voltage_penalty_weight":2.0}

input: "Clear sunny day with light winds."
output: {"event_type":"clear_sky","solar_scale":1.0,"wind_scale":0.6}

input: "Extreme heatwave expected this afternoon."
output: {"event_type":"heatwave","load_multiplier":1.4}

input: "Wind speeds are dropping across the region."
output: {"event_type":"low_wind","wind_scale":0.4}

input: "Natural gas supply disruption reported."
output: {"event_type":"fuel_disruption","gas_scale":0.5}

Output JSON only.
"""


# ----------------------------------------------------------------------
# Offline rule-based fallback (used when no OPENAI_API_KEY)
# ----------------------------------------------------------------------
def _rule_based_predict(text: str) -> dict[str, Any]:
    t = text.lower()
    out: dict[str, Any] = {}
    event_type = None

    # ---- weather: clouds / storm / clear ----
    if any(k in t for k in ("storm", "thunderstorm", "hurricane", "blizzard")):
        event_type = "storm"
        out["solar_scale"] = 0.3
        out["wind_scale"] = 1.2
        out["voltage_penalty_weight"] = 2.0
    elif any(k in t for k in ("cloud", "overcast", "cloudy")):
        event_type = "cloud_cover"
        out["solar_scale"] = 0.5
    elif any(k in t for k in ("clear sky", "sunny", "clear and sunny", "bright sun")):
        event_type = "clear_sky"
        out["solar_scale"] = 1.0

    # ---- wind ----
    if any(k in t for k in ("strong wind", "high wind", "gust", "windy")):
        event_type = event_type or "high_wind"
        out["wind_scale"] = 1.3
    elif any(k in t for k in ("calm", "low wind", "wind drop", "wind dropping",
                              "wind speeds are dropping", "light wind")):
        # Light wind shouldn't override an already-set storm event_type.
        event_type = event_type or "low_wind"
        # Light wind != dead calm. Use 0.6 unless it's clearly dropping/calm.
        if "light wind" in t and "drop" not in t:
            out["wind_scale"] = min(out.get("wind_scale", 1.0), 0.6)
        else:
            out["wind_scale"] = 0.4

    # ---- temperature / demand ----
    if any(k in t for k in ("heatwave", "heat wave", "extreme heat",
                            "very hot", "scorching")):
        event_type = event_type or "heatwave"
        out["load_multiplier"] = 1.4
    elif any(k in t for k in ("cold snap", "cold wave", "freeze", "freezing",
                              "extreme cold")):
        event_type = event_type or "cold_wave"
        out["load_multiplier"] = 1.5

    # ---- explicit percentage demand changes ----
    m = re.search(r"(?:demand|load|usage|consumption).*?(\d+)\s*(?:%|percent).*?(?:more|increase|higher)", t)
    if m:
        out["load_multiplier"] = max(0.5, min(2.0, 1.0 + int(m.group(1)) / 100.0))
    m = re.search(r"(?:demand|load|usage|consumption).*?(?:drop|less|decrease|reduce).*?(\d+)\s*(?:%|percent)", t)
    if m:
        out["load_multiplier"] = max(0.5, min(2.0, 1.0 - int(m.group(1)) / 100.0))

    # ---- gas / fuel ----
    if any(k in t for k in ("fuel disrupt", "gas disrupt", "fuel shortage",
                            "gas shortage", "pipeline", "fuel supply")):
        event_type = event_type or "fuel_disruption"
        out["gas_scale"] = 0.5

    # ---- grid emergency / equipment ----
    if any(k in t for k in ("grid emergency", "emergency", "blackout warning",
                            "grid alert")):
        event_type = event_type or "grid_emergency"
        out["voltage_penalty_weight"] = 3.0
    if any(k in t for k in ("equipment failure", "transformer fault",
                            "line trip", "outage")):
        event_type = event_type or "equipment_failure"
        out["voltage_penalty_weight"] = 3.0

    # ---- night ----
    if any(k in t for k in ("night", "nighttime", "after sunset", "midnight")):
        event_type = event_type or "cloud_cover"
        out["solar_scale"] = 0.0

    if event_type is None and not out:
        return {"event_type": "none", "status": "no_environment_change"}

    out["event_type"] = event_type or "none"
    return out


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def predict_event(event_description: str) -> dict[str, Any]:
    """Return validated JSON describing how the event affects the simulation."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": event_description},
                ],
            )
            raw = resp.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            return _validate(parsed)
        except Exception as e:  # noqa: BLE001
            parsed = _rule_based_predict(event_description)
            parsed["note"] = f"(LLM error: {e}; fell back to rules)"
            return _validate(parsed)
    return _validate(_rule_based_predict(event_description))


def _validate(parsed: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown keys, coerce numerics, clamp to spec ranges."""
    if not isinstance(parsed, dict):
        return {"event_type": "none", "status": "no_environment_change"}

    out: dict[str, Any] = {}

    et = parsed.get("event_type")
    if isinstance(et, str) and et in EVENT_TYPES:
        out["event_type"] = et
    else:
        out["event_type"] = "none"

    for k, (lo, hi) in _RANGES.items():
        if k not in parsed:
            continue
        try:
            v = float(parsed[k])
        except (TypeError, ValueError):
            continue
        out[k] = max(lo, min(hi, v))

    # Pass through informational fields
    if "status" in parsed and isinstance(parsed["status"], str):
        out["status"] = parsed["status"]
    if "note" in parsed and isinstance(parsed["note"], str):
        out["note"] = parsed["note"]

    # If no actionable parameter was produced, mark as no-change.
    actionable = set(_RANGES.keys()) & set(out.keys())
    if not actionable and out["event_type"] == "none":
        out.setdefault("status", "no_environment_change")

    return out


def apply_to_env(env, prediction: dict[str, Any]) -> dict[str, Any]:
    """Apply a validated prediction to a SmartGridEnv. Returns the params actually applied."""
    applied: dict[str, Any] = {}
    scale_kwargs: dict[str, float] = {}
    if "solar_scale" in prediction:
        scale_kwargs["solar_scale"] = float(prediction["solar_scale"])
    if "wind_scale" in prediction:
        scale_kwargs["wind_scale"] = float(prediction["wind_scale"])
    if "gas_scale" in prediction:
        scale_kwargs["gas_scale"] = float(prediction["gas_scale"])
    if "load_multiplier" in prediction:
        scale_kwargs["load_scale"] = float(prediction["load_multiplier"])
    if scale_kwargs:
        env.set_scales(**scale_kwargs)
        applied.update(scale_kwargs)

    rw_kwargs: dict[str, float] = {}
    if "voltage_penalty_weight" in prediction:
        rw_kwargs["voltage_penalty_weight"] = float(prediction["voltage_penalty_weight"])
    if "curtailment_penalty_weight" in prediction:
        # Env uses the historical name "curtailment_weight".
        rw_kwargs["curtailment_weight"] = float(prediction["curtailment_penalty_weight"])
    if rw_kwargs:
        env.set_reward_weights(**rw_kwargs)
        applied.update(rw_kwargs)

    return applied
