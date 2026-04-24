"""
Phase 2 — train a PPO agent on SmartGridEnv.

Usage (quick sanity training, ~1-2 min on CPU):
    python scripts/train_ppo.py --timesteps 20000

Full training (per the spec):
    python scripts/train_ppo.py --timesteps 1000000

The trained model is saved to `models/ppo_smartgrid.zip` by default.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.rl.env import EnvConfig, SmartGridEnv


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "ppo_smartgrid",
)


def make_env(seed: int = 0):
    def _thunk():
        env = SmartGridEnv(EnvConfig(seed=seed))
        return Monitor(env)
    return _thunk


def train(timesteps: int, model_path: str, seed: int = 0) -> str:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    vec_env = DummyVecEnv([make_env(seed)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=256,           # 256/env * 1 env => 256 steps per rollout
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        device="cpu",
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(model_path)
    print(f"\nSaved PPO model to: {model_path}.zip")
    return model_path


def evaluate(model_path: str, n_episodes: int = 3) -> None:
    model = PPO.load(model_path)
    env = SmartGridEnv(EnvConfig(seed=42))

    print(f"\n=== Evaluating {model_path} over {n_episodes} episodes ===")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=100 + ep)
        total = 0.0
        viol = 0
        for _ in range(24):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            if info["violated"]:
                viol += 1
            if term or trunc:
                break
        print(f"  ep {ep}: reward={total:>8.2f}  violations={viol:>2}/24")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=20_000,
                   help="PPO training timesteps (spec: 1_000_000).")
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    if not args.eval_only:
        train(args.timesteps, args.model_path, args.seed)
    evaluate(args.model_path)


if __name__ == "__main__":
    main()
