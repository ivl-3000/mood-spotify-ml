#!/usr/bin/env python3
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import create_engine, text


def setup_logging(level: str = "INFO"):
    logging.basicConfig(level=getattr(logging, level.upper()), format='%(asctime)s - %(levelname)s - %(message)s')


def run_clean(database_url: str) -> None:
    engine = create_engine(database_url, future=True)
    with engine.begin() as conn:
        # Remove social posts with neither title nor content
        conn.execute(text("""
            DELETE FROM social_posts
            WHERE (title IS NULL OR TRIM(title) = '')
              AND (content IS NULL OR TRIM(content) = '')
        """))
        
        # Remove audio_features whose track_id does not exist
        conn.execute(text("""
            DELETE FROM audio_features
            WHERE track_id NOT IN (SELECT id FROM tracks)
        """))
        
        # Remove playlist_tracks with missing refs
        conn.execute(text("""
            DELETE FROM playlist_tracks
            WHERE track_id NOT IN (SELECT id FROM tracks)
               OR playlist_id NOT IN (SELECT id FROM playlists)
        """))
        
        # Optional: trim excessive whitespace in playlist names
        conn.execute(text("""
            UPDATE playlists SET name = NULLIF(TRIM(name), '')
        """))
        conn.execute(text("""
            UPDATE tracks SET name = NULLIF(TRIM(name), '')
        """))
        
        # SQLite-specific optimizations
        if database_url.startswith("sqlite"):
            conn.execute(text("PRAGMA optimize"))
            conn.execute(text("VACUUM"))


def main():
    parser = argparse.ArgumentParser(description="Clean DB duplicates and nulls")
    parser.add_argument("--db", default="sqlite:///data/spotify.db", help="DATABASE_URL, default sqlite path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]) 
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_clean(args.db)


if __name__ == "__main__":
    main()
