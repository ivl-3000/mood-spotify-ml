#!/usr/bin/env python3
"""
CLI to match Spotify tracks to Genius songs (lyrics page URLs and metadata).
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from etl.genius_client import GeniusClient
from etl.lyrics_collector import LyricsCollector


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('lyrics_collection.log')]
    )


def main():
    parser = argparse.ArgumentParser(description="Collect lyrics matches for a playlist JSON")
    parser.add_argument("playlist_json", help="Path to playlist JSON file from Spotify collector")
    parser.add_argument("--output-dir", default="data/lyrics", help="Output directory for JSONL matches")
    parser.add_argument("--access-token", help="Genius access token (or set GENIUS_ACCESS_TOKEN)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")

    args = parser.parse_args()
    setup_logging(args.log_level)

    token = args.access_token or os.getenv("GENIUS_ACCESS_TOKEN")
    genius = GeniusClient(access_token=token)
    collector = LyricsCollector(genius)

    if not token:
        logging.getLogger(__name__).warning("No GENIUS_ACCESS_TOKEN provided; requests will fail.")

    out_path = collector.process_playlist_file(args.playlist_json, args.output_dir)
    logging.getLogger(__name__).info(f"Done. Output: {out_path}")


if __name__ == "__main__":
    main()
