# Mood-Aware Spotify Recommender

This project builds an end-to-end pipeline that recommends Spotify tracks tailored to a user's current mood using multimodal signals: listening history, playlist context, lyrics sentiment, and social sentiment trends.

## Quickstart

1) Create and activate a virtual environment
```powershell
. .\scripts\setup_env.ps1 -EnvName .venv
. .\.venv\Scripts\Activate.ps1
```

2) Verify core tooling
```powershell
python --version
pip --version
jupyter --version
```

## Tech Stack
- Python, PyTorch, Hugging Face, scikit-learn, pandas
- SQL (SQLite/Postgres via SQLAlchemy)
- Jupyter for exploration, Power BI for dashboards

## Repository Structure
- `docs/` project docs (charter, architecture, datasets, ERD/workflow)
- `scripts/` environment setup and utilities
- `data/` raw/processed datasets (gitignored)
- `models/`, `outputs/`, `reports/` artifacts (gitignored)

## Phase 1 Deliverables
- Project Charter (goals, scope, tools)
- Architecture draft, datasets definition, ERD/workflow
- Reproducible environment setup

## Status
- Repo initialized and public (`mood-spotify-ml`)
- Environment created and dependencies installed

## Next
- Draft system architecture and dataset definitions
- Implement initial ETL and mood-aware baseline
