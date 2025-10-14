# Datasets

## Overview
This project integrates three primary sources: Spotify metadata, lyrics, and social sentiment aggregates. Data are staged in a raw zone, cleaned, and stored in SQLite/Postgres for processing.

## Sources
- Spotify Metadata
  - Fields: tracks, artists, albums, playlists, audio features (tempo, energy, valence, etc.)
  - Access: Spotify Web API (for metadata/audio features) or curated public exports/playlists
  - Notes: Respect API rate limits; cache responses; do not redistribute proprietary audio.

- Lyrics Dataset
  - Fields: track_id, lyrics_text, language, source
  - Access: Public datasets or licensed providers; avoid scraping where prohibited
  - Notes: Ensure licensing permits text analysis; store only text and derived features.

- Social Sentiment Aggregates
  - Fields: time_window, topic/artist/track tags, sentiment score, volume
  - Access: Archival academic datasets or pre-aggregated open sources
  - Notes: Use only aggregate statistics (no PII); consider proxies if access is limited.

## Core Tables (proposed)
- `tracks`
  - id, name, album_id, artist_id, duration_ms, popularity, release_date
- `artists`
  - id, name, genres (array), followers
- `albums`
  - id, name, release_date, label
- `audio_features`
  - track_id, danceability, energy, valence, tempo, key, mode, loudness
- `playlists`
  - id, name, owner, num_followers
- `playlist_tracks`
  - playlist_id, track_id, added_at, position
- `lyrics`
  - track_id, language, lyrics_text (or reference), source
- `lyrics_nlp`
  - track_id, sentiment_score, emotion_probs (json), dominant_emotion
- `social_sentiment`
  - time_window, entity_type (artist/track), entity_id, sentiment_score, volume
- `user_interactions` (optional for offline eval)
  - user_id (hashed), track_id, interaction_type, timestamp

## Storage Plan
- Raw files: `data/raw/` (CSV/JSON/Parquet) [gitignored]
- Cleaned/processed: `data/processed/` (Parquet) [gitignored]
- Relational store: SQLite for local dev; upgrade to Postgres for scale

## Licensing & Compliance
- Track all dataset licenses in `docs/licenses/` (to be created).
- Do not redistribute licensed content; store only necessary text/features.
- Anonymize user identifiers where present.

## Volume & Performance (estimates)
- Tracks: 100k–1M
- Playlists: 10k–100k
- Lyrics coverage: 60–90% depending on catalog match
- Storage: 1–10 GB Parquet + relational indexes (dev scale)

## Ingestion Checklist
- Define playlist seeds and track fields
- Map IDs across metadata, audio features, and lyrics
- Validate language detection and text quality
- Backfill missing lyrics sentiment with fallbacks
