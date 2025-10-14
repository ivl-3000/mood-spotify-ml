# Mood-Aware Spotify Recommender

This project builds an end-to-end pipeline that recommends Spotify tracks tailored to a user's current mood using multimodal signals: listening history, playlist context, lyrics sentiment, and social sentiment trends.

## Problem Statement
Conventional recommenders emphasize long-term preferences and popularity, which underperform when a user’s transient mood shifts (e.g., upbeat vs. mellow). We aim to condition recommendations on mood inferred from lyrics sentiment/emotion and contextual signals, improving short-term relevance without sacrificing overall quality.

## Approach
- Ingest Spotify metadata, lyrics, and social sentiment aggregates via ETL.
- Run NLP models (Transformers) to infer sentiment/emotion for tracks and aggregate to playlist/user context.
- Build a baseline recommender (content/CF) and apply a mood-aware re-ranking layer that biases results toward the current mood.
- Evaluate with offline ranking metrics and expose outputs via a dashboard for exploration.

## Tools
- ML/NLP: Python, PyTorch, Hugging Face Transformers, scikit-learn, pandas
- Data: SQLAlchemy with SQLite/Postgres; Parquet/CSV for intermediate artifacts
- Visualization: Jupyter for exploration; Power BI for dashboards
- Dev: git/GitHub, virtualenv, Mermaid for architecture diagrams

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
