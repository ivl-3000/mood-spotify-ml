#!/usr/bin/env python3
"""
Script to collect Spotify playlist metadata.
"""
import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from etl.playlist_collector import PlaylistCollector
from etl.spotify_client import SpotifyClient


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('playlist_collection.log')
        ]
    )


def main():
    """Main function to collect playlist metadata."""
    parser = argparse.ArgumentParser(description="Collect Spotify playlist metadata")
    parser.add_argument("playlist_ids", nargs="+", help="Spotify playlist IDs to collect")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory for metadata files")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--client-id", help="Spotify client ID (or set SPOTIFY_CLIENT_ID env var)")
    parser.add_argument("--client-secret", help="Spotify client secret (or set SPOTIFY_CLIENT_SECRET env var)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Initialize Spotify client
    client = SpotifyClient(client_id=args.client_id, client_secret=args.client_secret)
    
    if not client.authenticate():
        logger.error("Failed to authenticate with Spotify. Check your credentials.")
        sys.exit(1)
    
    # Initialize playlist collector
    collector = PlaylistCollector(client)
    
    # Collect metadata for all playlists
    logger.info(f"Starting collection for {len(args.playlist_ids)} playlists")
    saved_files = collector.collect_multiple_playlists(args.playlist_ids, args.output_dir)
    
    logger.info(f"Collection complete. Saved {len(saved_files)} files:")
    for filepath in saved_files:
        logger.info(f"  - {filepath}")


if __name__ == "__main__":
    main()
