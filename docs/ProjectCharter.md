# Project Charter: Mood-Aware Spotify Recommender

## 1. Goal
Build an end-to-end pipeline that recommends Spotify tracks tailored to a user's current mood by leveraging multimodal signals: listening history, playlist context, lyrics sentiment, and social sentiment trends.

## 2. Problem Statement
Standard recommenders optimize for general similarity or popularity and rarely adapt to transient user mood. We aim to enhance recommendation relevance by conditioning on inferred mood signals derived from text (lyrics, social posts) and user interaction data.

## 3. Objectives
- Deliver a working MVP that produces mood-conditioned recommendations for a user profile.
- Provide transparent evaluation with offline metrics and a shareable interactive dashboard.
- Ensure reproducible data ingestion, feature extraction, model training/inference, and reporting.

## 4. Scope
In scope:
- ETL: ingest Spotify playlist metadata, track audio/lyrics features, and external sentiment sources (e.g., Reddit/Twitter aggregates where permissible).
- NLP: sentiment and emotion classification on lyrics; topic/mood taxonomy mapping.
- Recommender: baseline collaborative/content-based model with mood-aware re-ranking.
- Dashboard: Power BI (or alt.) to explore mood segments, model outputs, and KPIs.

Out of scope (Phase 1):
- Real-time personalization service; mobile app integration; large-scale A/B tests.

## 5. Deliverables (Phase 1)
- Repository and documentation scaffold.
- This Project Charter, system architecture draft, datasets definition, ERD/workflow diagrams.
- Environment setup files and initial CI-ready structure.

## 6. Success Metrics
- Data completeness: ≥90% of target tracks with lyrics sentiment labels.
- Model baseline: Recall@10 ≥ baseline CF; mood-aware uplift: ≥5% relative vs. non-mood model on mood-coherent queries.
- Reproducibility: End-to-end pipeline runnable via documented scripts.
- Usability: Dashboard answering key stakeholder questions about mood and recommendation quality.

## 7. Stakeholders
- Product/Research: Defines mood taxonomy, evaluation criteria.
- Engineering/ML: Builds ETL, NLP, recommender, and dashboard.
- End Users: Music listeners benefiting from mood-tailored recommendations.

## 8. Timeline (Phase 1: Oct 1–Oct 7)
- Oct 1: Kickoff, repo init, Project Charter.
- Oct 2: Environment and dependencies setup.
- Oct 3: Architecture design.
- Oct 4: README first draft.
- Oct 5: Literature review.
- Oct 6: Dataset definitions.
- Oct 7: ER diagram and workflow chart.

## 9. Risks & Mitigations
- Lyrics data availability/licensing: use public datasets or metadata with permissible use; fall back to proxy features if needed.
- Social data access limits: rely on aggregated/archival datasets; decouple pipeline from live APIs.
- Mood taxonomy ambiguity: adopt standard emotion sets (e.g., NRC, Ekman) and map to internal labels.
- Cold-start for new users: back off to content-based with mood priors.

## 10. Tools & Stack
- Data/ML: Python, PyTorch/TensorFlow, Hugging Face, scikit-learn, pandas.
- NLP: Transformers for sentiment/emotion; classical baselines for comparison.
- Storage/ETL: CSV/Parquet, SQLite/Postgres (local dev), simple orchestrations via scripts/Make.
- Visualization: Power BI; optional Jupyter for exploratory analysis.
- DevOps: git, GitHub; lint/format (ruff/black) later phases.

## 11. Ethical Considerations
- Avoid reinforcing sensitive inferences; restrict mood use to user-controlled contexts.
- Document dataset licenses and permissible use; exclude harmful content.

## 12. Acceptance
This charter defines Phase 1 scope and deliverables. Changes require explicit update in docs and alignment with stakeholders.


