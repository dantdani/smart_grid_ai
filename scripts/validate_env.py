"""
Phase 1 validation script.

Runs a full 24-step random rollout of `SmartGridEnv`, printing per-step
voltages, curtailment, and reward, plus a summary of observation/action shapes
and any voltage violations. If this script runs cleanly, Phase 1 is done and
Phase 2 (PPO training) can start.
"""
from __future__ import annotations

import os
import sys

# Allow running as `python scripts/validate_env.py` from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np

from backend.rl.env import EnvConfig, SmartGridEnv, V_MAX, V_MIN


def rollout(env: SmartGridEnv, label: str, action_fn) -> tuple[float, int]:
    obs, info = env.reset(seed=0)
    print(f"\n=== {label} ===")
    print(f"{'hr':>4} {'reward':>8} {'v_min':>6} {'v_max':>6} "
          f"{'viol':>4} {'curt_kW':>8} {'avail_kW':>9}")
    print("-" * 60)
    total_reward = 0.0
    violations = 0
    for _ in range(24):
        action = action_fn()
        obs, reward, term, trunc, info = env.step(action)
        v = info["voltages_pu"]
        total_reward += reward
        if info["violated"]:
            violations += 1
        print(f"{info['hour']:>4.1f} {reward:>8.2f} "
              f"{v.min():>6.3f} {v.max():>6.3f} "
              f"{int(info['violated']):>4d} "
              f"{float(np.sum(info['curtailed_kw'])):>8.2f} "
              f"{float(np.sum(info['available_solar_kw'])):>9.2f}")
        if term or trunc:
            break
    print("-" * 60)
    print(f"total reward        : {total_reward:.2f}")
    print(f"steps with violation: {violations} / 24")
    return total_reward, violations


def main() -> None:
    env = SmartGridEnv(EnvConfig(seed=0))

    print("Observation space:", env.observation_space)
    print("Action space:     ", env.action_space)
    assert env.observation_space.shape == (26,)
    assert env.action_space.shape == (5,)

    # Baseline: no curtailment -> should expose voltage instability
    zero_total, zero_viol = rollout(
        env, "Baseline: NO curtailment (expected to violate)",
        action_fn=lambda: np.zeros(env.action_space.shape, dtype=np.float32),
    )

    # Random actions: partial curtailment -> fewer violations, lost PV
    rng = np.random.default_rng(123)
    rand_total, rand_viol = rollout(
        env, "Random curtailment",
        action_fn=lambda: rng.uniform(0, 1, env.action_space.shape).astype(np.float32),
    )

    # Safe-but-wasteful: full curtailment -> no violations, no PV
    full_total, full_viol = rollout(
        env, "Full curtailment (safe but wastes all PV)",
        action_fn=lambda: np.ones(env.action_space.shape, dtype=np.float32),
    )

    print(f"\nsafe band: [{V_MIN}, {V_MAX}] p.u.")
    print(f"no-curt violations  : {zero_viol} / 24  (must be > 0 for RL to learn)")
    print(f"random violations   : {rand_viol} / 24")
    print(f"full-curt violations: {full_viol} / 24  (must be 0)")
    ok = zero_viol > 0 and full_viol == 0
    print("\nPhase 1 validation OK." if ok else "\nFeeder needs retuning.")


if __name__ == "__main__":
    main()
