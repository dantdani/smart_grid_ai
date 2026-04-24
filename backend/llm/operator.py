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
    "load_scale",
    "voltage_penalty_weight",
    "curtailment_weight",
    "stability_bonus",
    "note",
}


_SYSTEM_PROMPT = """You translate grid-operator natural-language commands into JSON.

Return ONLY a JSON object with a subset of these keys (omit keys that the
command does not mention):

- solar_scale            (float 0..2): multiplier on available solar PV.
                         "solar drops 70%" -> 0.3; "solar doubles" -> 2.0.
- load_scale             (float 0..2): multiplier on residential load.
- voltage_penalty_weight (float, typical 0.5..5): higher = prioritize voltage safety.
- curtailment_weight     (float, typical 0.01..1): higher = preserve more PV.
- stability_bonus        (float, typical 1..20): reward when all voltages are safe.
- note                   (string): brief echo of the operator intent.

Examples:
"A storm is coming and solar generation will drop by 70 percent."
 -> {"solar_scale": 0.3, "note": "Storm: solar reduced by 70%."}

"Prioritize grid stability over renewable energy output."
 -> {"voltage_penalty_weight": 2.0, "curtailment_weight": 0.05,
     "note": "Stability prioritized over renewables."}

"Heatwave — households will use 50% more power."
 -> {"load_scale": 1.5, "note": "Heatwave: +50% load."}

Output JSON only, no prose."""


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
    """Parse a natural-language command. Uses OpenAI if configured, else regex."""
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
                    {"role": "user", "content": command},
                ],
            )
            raw = resp.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            return _clean(parsed)
        except Exception as e:  # noqa: BLE001
            fallback = _regex_fallback(command)
            fallback["note"] = f"(LLM error: {e}) " + fallback.get("note", "")
            return fallback
    return _regex_fallback(command)


def _clean(parsed: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in parsed.items():
        if k not in SCHEMA_KEYS:
            continue
        if k == "note":
            out[k] = str(v)
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
    # Clamp to reasonable ranges
    if "solar_scale" in out:
        out["solar_scale"] = max(0.0, min(2.0, out["solar_scale"]))
    if "load_scale" in out:
        out["load_scale"] = max(0.0, min(2.0, out["load_scale"]))
    for w in ("voltage_penalty_weight", "curtailment_weight", "stability_bonus"):
        if w in out:
            out[w] = max(0.0, out[w])
    return out
