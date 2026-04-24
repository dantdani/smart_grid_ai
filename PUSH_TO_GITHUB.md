# Push to GitHub

The local repo is ready. To publish to GitHub, run **one** of the following from
this directory (`smart_grid_ai/`):

## Option A — GitHub CLI (recommended)

```powershell
gh auth login                   # follow the prompts (browser)
gh repo create smart_grid_ai --public --source=. --remote=origin --push `
    --description "Hybrid agentic smart-grid simulation: PPO + pandapower OPF + LLM operator + React dashboard"
```

## Option B — Manual remote

1. Create an empty repo at https://github.com/new (no README, no .gitignore, no license).
2. Then:

```powershell
git remote add origin https://github.com/<your-username>/smart_grid_ai.git
git push -u origin main
```

The first commit (`9c1e80c`) already contains all phases 1-6 and the trained
PPO model artifact (`models/ppo_smartgrid.zip`).
