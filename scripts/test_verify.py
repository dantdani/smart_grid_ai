"""Phase 3 smoke test: drive env to an over-voltage scenario at mid-day, then
check that verify_action() falls through from RL (unsafe) to OPF override."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np

from backend.rl.env import EnvConfig, SmartGridEnv, V_MAX, V_MIN
from backend.safety.verify import verify_action


def main() -> None:
    # Start at hour 12 so first step is the mid-day PV peak → baseline violates.
    env = SmartGridEnv(EnvConfig(seed=0, start_hour=12))
    env.reset(seed=0)

    # Force a zero-curtailment action (the baseline we know violates).
    rl_action = np.zeros(5, dtype=np.float32)

    # Advance one env.step with no curtailment to populate _last_* fields if needed,
    # but actually reset() already fills them. Just call verify directly.
    verdict = verify_action(env, rl_action)

    print(f"source             = {verdict.source}")
    print(f"rl_violated        = {verdict.rl_violated}")
    print(f"opf_used           = {verdict.opf_used}")
    print(f"rl_curtailment     = {[round(x,3) for x in verdict.rl_curtailment]}")
    print(f"applied_curtailment= {[round(x,3) for x in verdict.applied_curtailment]}")
    print(f"rl_voltages        = {[round(x,4) for x in (verdict.rl_voltages or [])]}")
    print(f"applied_voltages   = {[round(x,4) for x in verdict.predicted_voltages]}")
    print(f"safe band          = [{V_MIN}, {V_MAX}]")
    print(f"message            = {verdict.message}")

    # Programmatic assertions.
    max_v_rl = max(verdict.rl_voltages) if verdict.rl_voltages else 0.0
    assert verdict.rl_violated, f"Expected RL=zero-curtail to violate; max v_rl={max_v_rl}"
    assert verdict.source in ("opf", "fallback"), f"Expected OPF/fallback override, got {verdict.source}"
    if verdict.source == "opf":
        v = np.array(verdict.predicted_voltages)
        assert np.all(v >= V_MIN - 1e-6) and np.all(v <= V_MAX + 1e-6), f"OPF result out of band: {v}"
        print("\nOK — Phase 3: OPF override enforced the safe voltage band.")
    else:
        print("\nOK — Phase 3: fallback curtailment applied (OPF infeasible).")


if __name__ == "__main__":
    main()
