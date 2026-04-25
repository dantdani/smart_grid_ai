"""
Self-correcting RL module.

Captures every RL action that fails physics verification along with the
OPF-derived safe action. The resulting (state, rl_action, opf_action,
timestamp) records form a supervised dataset that the live PPO policy can
be fine-tuned against, gradually pulling its behaviour toward the OPF
solution.

Public surface:
    CorrectionMemory      - bounded JSONL-backed buffer
    SelfCorrectingTrainer - records corrections, computes metrics, retrains
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Deque, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------
@dataclass
class CorrectionRecord:
    state: list[float]
    rl_action: list[float]
    opf_action: list[float]
    timestamp: int          # simulation step
    wall_time: float        # unix seconds
    voltage_min: float
    voltage_max: float
    source: str             # "opf" or "fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "rl_action": self.rl_action,
            "opf_action": self.opf_action,
            "timestamp": self.timestamp,
            "wall_time": self.wall_time,
            "voltage_min": self.voltage_min,
            "voltage_max": self.voltage_max,
            "source": self.source,
        }


class CorrectionMemory:
    """Bounded ring buffer of OPF corrections, persisted as JSONL on disk."""

    def __init__(self, capacity: int = 10_000, path: str | None = None) -> None:
        self.capacity = int(capacity)
        self.path = path
        self._buf: Deque[CorrectionRecord] = deque(maxlen=self.capacity)
        self._new_since_train = 0
        self._lock = Lock()
        if self.path and os.path.exists(self.path):
            self._load_from_disk()

    # ---- IO -----------------------------------------------------------
    def _load_from_disk(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        self._buf.append(CorrectionRecord(
                            state=list(d["state"]),
                            rl_action=list(d["rl_action"]),
                            opf_action=list(d["opf_action"]),
                            timestamp=int(d["timestamp"]),
                            wall_time=float(d.get("wall_time", 0.0)),
                            voltage_min=float(d.get("voltage_min", 0.0)),
                            voltage_max=float(d.get("voltage_max", 0.0)),
                            source=str(d.get("source", "opf")),
                        ))
                    except Exception:
                        continue
        except FileNotFoundError:
            pass

    def _append_to_disk(self, rec: CorrectionRecord) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")

    # ---- API ----------------------------------------------------------
    def add(self, rec: CorrectionRecord) -> None:
        with self._lock:
            self._buf.append(rec)
            self._new_since_train += 1
            self._append_to_disk(rec)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def new_since_train(self) -> int:
        return self._new_since_train

    def reset_new_counter(self) -> None:
        with self._lock:
            self._new_since_train = 0

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (states, rl_actions, opf_actions) as float32 ndarrays."""
        with self._lock:
            recs = list(self._buf)
        if not recs:
            return (np.zeros((0, 0), dtype=np.float32),
                    np.zeros((0, 0), dtype=np.float32),
                    np.zeros((0, 0), dtype=np.float32))
        states = np.asarray([r.state for r in recs], dtype=np.float32)
        rl = np.asarray([r.rl_action for r in recs], dtype=np.float32)
        opf = np.asarray([r.opf_action for r in recs], dtype=np.float32)
        return states, rl, opf

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in list(self._buf)[-n:]]

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._new_since_train = 0
        if self.path and os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
@dataclass
class CorrectionMetrics:
    rl_accepted: int = 0
    rl_rejected: int = 0
    opf_corrections: int = 0
    fallback_used: int = 0
    voltage_violations: int = 0
    total_steps: int = 0
    last_retrain_step: int = 0
    last_retrain_loss: float | None = None
    retrain_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def acceptance_rate(self) -> float:
        denom = self.rl_accepted + self.rl_rejected
        return float(self.rl_accepted / denom) if denom else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rl_accepted": self.rl_accepted,
            "rl_rejected": self.rl_rejected,
            "opf_corrections": self.opf_corrections,
            "fallback_used": self.fallback_used,
            "voltage_violations": self.voltage_violations,
            "total_steps": self.total_steps,
            "acceptance_rate": self.acceptance_rate(),
            "last_retrain_step": self.last_retrain_step,
            "last_retrain_loss": self.last_retrain_loss,
            "retrain_count": self.retrain_count,
            "buffer_size": None,  # filled in by trainer
        }


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------
class SelfCorrectingTrainer:
    """Wraps a CorrectionMemory + metrics + a behaviour-cloning retrainer.

    The retrainer fine-tunes the SB3 PPO policy against (state -> opf_action)
    pairs using a simple supervised MSE loss on the policy's mean action.
    """

    def __init__(
        self,
        capacity: int = 10_000,
        memory_path: str | None = None,
        retrain_every_steps: int = 1000,
        retrain_min_new: int = 500,
        retrain_epochs: int = 4,
        retrain_batch_size: int = 64,
        retrain_lr: float = 3e-4,
    ) -> None:
        self.memory = CorrectionMemory(capacity=capacity, path=memory_path)
        self.metrics = CorrectionMetrics()
        self.retrain_every_steps = int(retrain_every_steps)
        self.retrain_min_new = int(retrain_min_new)
        self.retrain_epochs = int(retrain_epochs)
        self.retrain_batch_size = int(retrain_batch_size)
        self.retrain_lr = float(retrain_lr)

    # ---- recording ----------------------------------------------------
    def record_step(
        self,
        *,
        obs: np.ndarray,
        rl_action: np.ndarray,
        applied_action: np.ndarray,
        source: str,
        rl_violated: bool,
        info_violated: bool,
        timestep: int,
        voltages: Iterable[float],
    ) -> dict[str, Any] | None:
        """Update metrics and (if applicable) store an OPF correction.

        Returns the JSON log entry for the correction (or None).
        """
        self.metrics.total_steps += 1

        v = np.asarray(list(voltages), dtype=float)
        v_min = float(v.min()) if v.size else 0.0
        v_max = float(v.max()) if v.size else 0.0

        if info_violated:
            self.metrics.voltage_violations += 1

        if source == "rl":
            self.metrics.rl_accepted += 1
            return None

        # source is "opf" or "fallback" — RL was rejected.
        self.metrics.rl_rejected += 1
        if source == "opf":
            self.metrics.opf_corrections += 1
        else:
            self.metrics.fallback_used += 1

        rec = CorrectionRecord(
            state=[float(x) for x in np.asarray(obs).reshape(-1)],
            rl_action=[float(x) for x in np.asarray(rl_action).reshape(-1)],
            opf_action=[float(x) for x in np.asarray(applied_action).reshape(-1)],
            timestamp=int(timestep),
            wall_time=time.time(),
            voltage_min=v_min,
            voltage_max=v_max,
            source=source,
        )
        self.memory.add(rec)

        return {
            "event": "rl_correction",
            "state": rec.state,
            "rl_action": rec.rl_action,
            "opf_action": rec.opf_action,
            "timestamp": rec.timestamp,
            "source": rec.source,
            "voltage_min": rec.voltage_min,
            "voltage_max": rec.voltage_max,
        }

    # ---- retraining ---------------------------------------------------
    def should_retrain(self) -> bool:
        if len(self.memory) < self.retrain_batch_size:
            return False
        if self.memory.new_since_train >= self.retrain_min_new:
            return True
        steps_since = self.metrics.total_steps - self.metrics.last_retrain_step
        if steps_since >= self.retrain_every_steps and self.memory.new_since_train > 0:
            return True
        return False

    def retrain(self, sb3_model) -> dict[str, Any]:
        """Behaviour-cloning fine-tune of the PPO policy on stored corrections.

        `sb3_model` is a stable_baselines3.PPO instance whose `.policy` is a
        torch nn.Module. Returns a dict of loss / metadata.
        """
        states, _rl, opf = self.memory.as_arrays()
        if states.shape[0] == 0 or sb3_model is None:
            return {"status": "skipped", "reason": "no data or no model"}

        try:
            import torch
            import torch.nn.functional as F
        except Exception as e:  # noqa: BLE001
            return {"status": "skipped", "reason": f"torch unavailable: {e}"}

        policy = sb3_model.policy
        device = policy.device
        s_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        a_t = torch.as_tensor(opf, dtype=torch.float32, device=device)

        optim = torch.optim.Adam(policy.parameters(), lr=self.retrain_lr)
        n = s_t.shape[0]
        bs = min(self.retrain_batch_size, n)
        last_loss = float("nan")

        policy.train()
        for _epoch in range(self.retrain_epochs):
            perm = torch.randperm(n, device=device)
            losses: list[float] = []
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                obs_b = s_t[idx]
                tgt_b = a_t[idx]

                # Get the policy's mean action. SB3 PPO with a Box action
                # space uses a Gaussian whose mean is the network output.
                dist = policy.get_distribution(obs_b)
                if hasattr(dist, "distribution") and hasattr(dist.distribution, "mean"):
                    mean = dist.distribution.mean
                else:
                    # Fallback: deterministic forward
                    mean, _, _ = policy(obs_b, deterministic=True)
                # PPO often parameterises actions in an unbounded space and
                # the env clips at [0,1]; we BC against the clipped range.
                pred = torch.clamp(mean, 0.0, 1.0)
                tgt = torch.clamp(tgt_b, 0.0, 1.0)
                loss = F.mse_loss(pred, tgt)
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(float(loss.detach().cpu()))
            if losses:
                last_loss = float(np.mean(losses))
        policy.eval()

        self.memory.reset_new_counter()
        self.metrics.last_retrain_step = self.metrics.total_steps
        self.metrics.last_retrain_loss = last_loss
        self.metrics.retrain_count += 1
        self.metrics.history.append({
            "step": self.metrics.total_steps,
            "loss": last_loss,
            "n_samples": int(n),
        })

        return {
            "status": "ok",
            "loss": last_loss,
            "samples": int(n),
            "epochs": self.retrain_epochs,
            "retrain_count": self.metrics.retrain_count,
        }

    # ---- introspection ------------------------------------------------
    def status(self) -> dict[str, Any]:
        d = self.metrics.to_dict()
        d["buffer_size"] = len(self.memory)
        d["buffer_capacity"] = self.memory.capacity
        d["new_since_train"] = self.memory.new_since_train
        d["should_retrain"] = self.should_retrain()
        return d
