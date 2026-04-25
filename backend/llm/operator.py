"""
Phase 5 — LLM operator command interface.

Translates natural-language operator commands into structured grid/reward
overrides. Falls back to a regex-based parser when no OPENAI_API_KEY is set
so the demo still works fully offline.

Schema returned to the API layer (any subset of keys may be present):
    {
      "solar_scale":             float in [0, 2],   # multiplies available PV
      "load_scale":              float in [0, 2],   # multiplies residential load
      "voltage_penalty_weight":  float > 0,         # RL reward weight
      "curtailment_weight":      float >= 0,        # RL reward weight
      "stability_bonus":         float >= 0,        # RL reward weight
      "note":                    str                # human-readable echo
    }
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

SCHEMA_KEYS = {
    "solar_scale",
    "wind_scale",
    "gas_scale",
    "load_scale",
    "voltage_penalty_weight",
    "curtailment_weight",
    "stability_bonus",
    "note",
    "reasoning",
}


_SYSTEM_PROMPT = """You translate grid-operator natural-language commands into JSON.

You MUST return a single JSON object with TWO top-level fields:

1. "decision": an object containing a subset of these keys (omit keys that the
   command does not mention):
     - solar_scale            (float 0..2): multiplier on the 3 solar DERs' availability.
                              "solar drops 70%" -> 0.3; "solar doubles" -> 2.0.
     - wind_scale             (float 0..2): multiplier on the 1 wind DER's availability.
                              "strong winds" / "windy" -> 1.5..2.0; "calm winds" -> 0.3..0.7.
     - gas_scale              (float 0..2): multiplier on the 1 gas DER's dispatchable cap.
                              "gas plant offline" -> 0; "reserve more gas" -> 1.5.
     - load_scale             (float 0..2): multiplier on residential load.
     - voltage_penalty_weight (float, typical 0.5..5): higher = prioritize voltage safety.
     - curtailment_weight     (float, typical 0.01..1): higher = preserve more PV/wind.
     - stability_bonus        (float, typical 1..20): reward when all voltages are safe.
     - note                   (string): brief echo of the operator intent.

2. "reasoning": a short 1-3 sentence string explaining WHY you chose those
   values. Reference the operator's intent, the physical effect on the grid
   (10 houses, 5 DERs: 3 solar + 1 wind + 1 gas, voltage band 0.95-1.05 pu),
   and any trade-offs made.

ALWAYS produce the "decision" object first. The reasoning is for the human
operator to review; it must NOT change the decision values.

Examples:

Input: "A storm is coming and solar generation will drop by 70 percent."
Output:
{"decision": {"solar_scale": 0.3, "note": "Storm: solar reduced by 70%."},
 "reasoning": "The storm cuts incoming irradiance, so solar_scale=0.3 reflects a 70% drop in PV availability. Wind/gas DERs and reward weights are left untouched because the operator only described a solar event."}

Input: "Prioritize grid stability over renewable energy output."
Output:
{"decision": {"voltage_penalty_weight": 2.0, "curtailment_weight": 0.05,
              "note": "Stability prioritized over renewables."},
 "reasoning": "Doubling voltage_penalty_weight to 2.0 makes the RL agent more aggressive about keeping voltages inside 0.95-1.05 pu, while shrinking curtailment_weight to 0.05 makes it cheap to curtail solar/wind when needed for stability."}

Input: "Heatwave - households will use 50% more power."
Output:
{"decision": {"load_scale": 1.5, "note": "Heatwave: +50% load."},
 "reasoning": "A 50% load surge across 10 houses risks under-voltage at the feeder tail; load_scale=1.5 simulates this so the RL/OPF stack can react. Reward weights are kept at defaults so the agent's normal trade-offs apply."}

Input: "Strong winds are expected from the east."
Output:
{"decision": {"wind_scale": 1.8, "note": "Strong winds: wind DER boosted."},
 "reasoning": "Strong winds raise the wind turbine's available output, so wind_scale=1.8 boosts the single wind DER. Solar, gas, and reward weights are unchanged because the operator only flagged a wind event."}

Output JSON only, no prose, no code fences."""


def _regex_fallback(command: str) -> dict[str, Any]:
    """Offline parser covering the spec's example commands."""
    cmd = command.lower()
    out: dict[str, Any] = {}

    # "solar ... drop/reduce/fall by X percent"
    m = re.search(r"solar.*?(?:drop|reduce|fall|decrease).*?(\d+)\s*(?:%|percent)", cmd)
    if m:
        out["solar_scale"] = max(0.0, 1.0 - int(m.group(1)) / 100.0)

    # "solar ... increase/boost/double by X percent"
    m = re.search(r"solar.*?(?:increase|rise|boost|grow).*?(\d+)\s*(?:%|percent)", cmd)
    if m:
        out["solar_scale"] = min(2.0, 1.0 + int(m.group(1)) / 100.0)

    if "double" in cmd and "solar" in cmd:
        out["solar_scale"] = 2.0
    if ("no solar" in cmd) or ("solar off" in cmd):
        out["solar_scale"] = 0.0

    # Wind scaling
    if re.search(r"\b(strong|high|gust\w*|heavy)\b.*\bwind", cmd) or re.search(r"\bwind\w*\b.*\b(strong|pick\w* up|increas\w*|gust\w*)\b", cmd):
        out["wind_scale"] = 1.8
    elif re.search(r"\b(calm|light|low|no|drop\w*|weak)\b.*\bwind", cmd) or re.search(r"\bwind\w*\b.*\b(calm|drop\w*|die\w*|weak)\b", cmd):
        out["wind_scale"] = 0.3
    m = re.search(r"wind.*?(?:drop|reduce|fall|decrease).*?(\d+)\s*(?:%|percent)", cmd)
    if m:
        out["wind_scale"] = max(0.0, 1.0 - int(m.group(1)) / 100.0)
    m = re.search(r"wind.*?(?:increase|rise|boost|grow).*?(\d+)\s*(?:%|percent)", cmd)
    if m:
        out["wind_scale"] = min(2.0, 1.0 + int(m.group(1)) / 100.0)

    # Gas scaling
    if re.search(r"\bgas\b.*\b(off|offline|down|trip\w*|outage)\b", cmd):
        out["gas_scale"] = 0.0
    elif re.search(r"\b(reserve|boost|more)\b.*\bgas\b", cmd) or re.search(r"\bgas\b.*\b(boost\w*|reserve\w*|increase\w*)\b", cmd):
        out["gas_scale"] = 1.5

    # Load scaling
    m = re.search(r"(?:load|demand|usage|households?).*?(?:increase|more).*?(\d+)\s*(?:%|percent)", cmd)
    if m:
        out["load_scale"] = min(2.0, 1.0 + int(m.group(1)) / 100.0)
    m = re.search(r"(?:load|demand|usage).*?(?:drop|reduce|less).*?(\d+)\s*(?:%|percent)", cmd)
    if m:
        out["load_scale"] = max(0.0, 1.0 - int(m.group(1)) / 100.0)

    # Priorities
    if "prioritize" in cmd or "priorit" in cmd:
        if "stability" in cmd or "safety" in cmd or "voltage" in cmd:
            out["voltage_penalty_weight"] = 2.0
            out["curtailment_weight"] = 0.05
        elif "renewable" in cmd or "solar" in cmd or "green" in cmd:
            out["voltage_penalty_weight"] = 0.5
            out["curtailment_weight"] = 0.3

    if not out:
        out["note"] = f"No structured override extracted from: {command!r}"
    else:
        out.setdefault("note", command.strip())
    return out


def parse_operator_command(command: str) -> dict[str, Any]:
    """Parse a natural-language command. Uses OpenAI if configured, else regex.

    Returns a flat dict containing any of the SCHEMA_KEYS plus an optional
    "reasoning" string explaining why those values were chosen. The decision
    fields are extracted first and validated independently of the reasoning,
    so a malformed reasoning never corrupts the applied parameters.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            model = os.environ.get("OPENAI_MODEL", "gpt-5")
            kwargs: dict[str, Any] = {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": command},
                ],
            }
            # GPT-5 / o-series reject custom temperature; only set it for older
            # chat models that accept it.
            if not (model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")):
                kwargs["temperature"] = 0.0
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or "{}"
            parsed = json.loads(raw)

            # Accept either {"decision": {...}, "reasoning": "..."} (preferred)
            # or a flat dict for backward compatibility.
            if isinstance(parsed, dict) and isinstance(parsed.get("decision"), dict):
                decision = _clean(parsed["decision"])
                reasoning = parsed.get("reasoning")
                if reasoning:
                    decision["reasoning"] = str(reasoning)
                return decision
            return _clean(parsed)
        except Exception as e:  # noqa: BLE001
            fallback = _regex_fallback(command)
            fallback["reasoning"] = (
                f"(LLM error: {e}) Fell back to offline regex parser."
            )
            return fallback
    out = _regex_fallback(command)
    out.setdefault("reasoning", "Offline regex parser (no OPENAI_API_KEY set).")
    return out


def _clean(parsed: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in parsed.items():
        if k not in SCHEMA_KEYS:
            continue
        if k in ("note", "reasoning"):
            out[k] = str(v)
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
    # Clamp to reasonable ranges
    for k in ("solar_scale", "wind_scale", "gas_scale", "load_scale"):
        if k in out:
            out[k] = max(0.0, min(2.0, out[k]))
    for w in ("voltage_penalty_weight", "curtailment_weight", "stability_bonus"):
        if w in out:
            out[w] = max(0.0, out[w])
    return out
