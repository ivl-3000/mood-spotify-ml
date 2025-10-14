# System Architecture

This architecture implements an end-to-end pipeline from ETL to NLP feature extraction, mood-aware recommender, and reporting/dashboard.

![ETL -> NLP -> Recommender -> Dashboard](docs/diagrams/img/mood_reco_workflow.png)

Source (Mermaid): `docs/diagrams/mood_reco_workflow.mmd`

## Realistic Spotify-style Platform Architecture

![Spotify-style platform architecture](docs/diagrams/img/spotify_system.png)

Source (Mermaid): `docs/diagrams/spotify_system.mmd`

## Images
- Spotify-style platform (PNG): `docs/diagrams/img/spotify_system.png`
- Mood-aware workflow (PNG): `docs/diagrams/img/mood_reco_workflow.png`

## Components
- ETL: Batch ingestion of Spotify metadata, lyrics, and social aggregates into a raw zone; cleaning and normalization into `SQLite/Postgres`.
- NLP: Hugging Face transformers for sentiment/emotion classification on lyrics; aggregation to track and playlist mood features.
- Feature Engineering: Content features (audio, lyrics), interaction features, and CF-ready matrices.
- Recommender: Baseline CF/content model; mood-aware re-ranking that conditions on current mood features.
- Serving/Analysis: Batch notebooks or lightweight API for inference; results stored for reporting.
- Dashboard: Power BI for mood segments, recommendation quality, and KPIs.

## Storage
- Raw data: `data/` (gitignored)
- Processed tables: `SQLite/Postgres` via SQLAlchemy
- Artifacts: `models/`, `outputs/`, `reports/` (gitignored)

## Orchestration
- Phase 1 uses scripts and notebooks; migrate to lightweight orchestration later if needed.
