"""
Lyrics collector: match Spotify tracks to Genius songs and store metadata.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from .genius_client import GeniusClient


class LyricsCollector:
    def __init__(self, genius: Optional[GeniusClient] = None) -> None:
        self.genius = genius or GeniusClient()
        self.logger = logging.getLogger(__name__)

    def match_track(self, track: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Given a Spotify track object from our playlist JSON, search Genius and return
        a minimal record including song id, url, and title if matched.
        """
        name = track.get("name") or ""
        artists = track.get("artists") or []
        primary_artist = artists[0]["name"] if artists else ""
        query = f"{name} {primary_artist}"
        search = self.genius.search_song(query)
        if not search:
            return None
        best = GeniusClient.best_match(search, name, primary_artist)
        if not best:
            return None
        return {
            "genius_song_id": best.get("id"),
            "genius_url": best.get("url"),
            "full_title": best.get("full_title"),
            "primary_artist": best.get("primary_artist", {}).get("name"),
        }

    def process_playlist_file(self, playlist_json_path: str, output_dir: str = "data/lyrics") -> str:
        """
        For a saved playlist JSON (from our ETL), attempt to match each track to Genius.
        Stores a JSONL file where each line is a merged record of track + genius fields.
        """
        path = Path(playlist_json_path)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        out_path = output / (path.stem + "_lyrics.jsonl")

        with open(path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
            playlist = json.load(f_in)
            for t in playlist.get("tracks", []):
                match = self.match_track(t)
                record = {
                    "playlist_id": playlist.get("playlist_id"),
                    "track_id": t.get("track_id"),
                    "track_name": t.get("name"),
                    "artists": [a.get("name") for a in t.get("artists", [])],
                    "album_id": t.get("album", {}).get("id"),
                    "genius": match or None,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.logger.info(f"Wrote lyrics matches to {out_path}")
        return str(out_path)
