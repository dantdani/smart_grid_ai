# Smart Grid AI — Hybrid Agentic Control Platform

Web-based simulation of a 10-house residential feeder with 5 rooftop-solar
DERs, controlled by a three-layer agentic stack.

## Three-layer control stack

1. **RL decision layer (PPO)** — proposes per-DER solar curtailment.
2. **Physics verification (pandapower)** — runs power flow on the proposed
   action; if any bus leaves `[0.95, 1.05] p.u.`, runs OPF and/or falls back
   to full curtailment. The unsafe RL action is never applied.
3. **LLM operator interface** — plain-English commands
   (`"storm coming, solar drops 70%"`, `"prioritize stability"`) are parsed
   to structured overrides (`solar_scale`, `load_scale`, reward weights).
   Uses OpenAI if `OPENAI_API_KEY` is set, otherwise a regex fallback.

## Phase status

| Phase | Component                          | Status |
|-------|------------------------------------|--------|
| 1     | pandapower grid + Gym env          | ✅     |
| 2     | PPO training + eval                | ✅     |
| 3     | OPF safety verification            | ✅     |
| 4     | FastAPI backend + WebSocket        | ✅     |
| 5     | LLM operator command parser        | ✅     |
| 6     | React dashboard (CDN, zero-build)  | ✅     |

## Folder structure

```
smart_grid_ai/
├── backend/
│   ├── grid/       pandapower feeder + load/solar profiles
│   ├── rl/env.py   SmartGridEnv (obs=26, action=5)
│   ├── safety/     verify_action() with pandapower OPF
│   ├── llm/        parse_operator_command() (OpenAI + regex fallback)
│   └── api/        FastAPI app (server.py)
├── scripts/
│   ├── validate_env.py   Phase 1 smoke test
│   ├── train_ppo.py      Phase 2: train + eval PPO
│   └── test_verify.py    Phase 3: OPF override smoke test
├── frontend/index.html   Phase 6: React + D3 + Recharts dashboard
├── models/ppo_smartgrid.zip  (created after training)
├── requirements.txt
└── README.md
```

## Install

```powershell
cd smart_grid_ai
python -m pip install -r requirements.txt
```

## Run

### 1. Validate the physics env (Phase 1)
```powershell
python scripts/validate_env.py
```
Expected: `Phase 1 validation OK.` (baseline violates, full-curtailment safe).

### 2. Train PPO (Phase 2)
```powershell
python scripts/train_ppo.py --timesteps 2048      # smoke (~1 min)
python scripts/train_ppo.py --timesteps 1000000   # full spec run
```
Saves `models/ppo_smartgrid.zip` and evaluates 3 episodes.

### 3. Test the OPF safety layer (Phase 3)
```powershell
python scripts/test_verify.py
```
At `start_hour=12` the zero-curtailment action drives v_max above 1.05; the
verifier must route to either OPF or full-curtailment fallback.

### 4. Launch backend + dashboard (Phases 4–6)
```powershell
# terminal A — FastAPI backend
python -m uvicorn backend.api.server:app --host 127.0.0.1 --port 8000

# terminal B — open the dashboard
start frontend/index.html
```
The dashboard provides:
- Radial feeder map (D3) colored by voltage, DER houses marked ☼.
- Live telemetry (Recharts): voltage envelope vs. the safe band; solar
  availability vs. delivered / curtailed.
- Operator console: plain-English command box.
- Decision log: per-step RL / OPF / fallback tagging.

### 5. LLM operator — example commands

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/operator_command -Method Post `
  -ContentType "application/json" `
  -Body '{"command":"A storm is coming and solar generation will drop by 70 percent"}'
```
Parses to `{"solar_scale": 0.3, ...}`. Other examples: `"prioritize stability
over yield"`, `"heatwave, loads up 30%"`, `"no solar today"`.

## Environment constants

| Quantity        | Value |
|-----------------|-------|
| houses          | 10 (radial) |
| DER houses      | `[1,3,5,7,9]` |
| PV capacity     | 5 kW / DER |
| feeder segment  | 80 m, 0.9 Ω/km, 0.09 Ω/km X |
| slack voltage   | 1.025 p.u. |
| safe band       | `[0.95, 1.05]` p.u. |
| observation     | 26 (10 V + 10 P_load + 5 P_avail + 1 t/24) |
| action          | 5 curtailment ratios ∈ `[0, 1]` |

## Notes

- `numba` is not required; pandapower prints a warning but still runs.
- The PPO model is optional: without it the backend uses a zero-curtailment
  proposal, which still exercises the OPF/fallback layer end-to-end.
- Set `OPENAI_API_KEY` to route operator commands through `gpt-4o-mini`;
  otherwise a deterministic regex fallback is used.
