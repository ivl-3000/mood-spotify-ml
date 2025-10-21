"""
Genius API client for lyrics metadata.
"""
from __future__ import annotations
import os
import logging
from typing import Dict, Optional, Any, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class GeniusClient:
    """Client for interacting with Genius API (search + song metadata).
    Note: The official API does not return full lyrics; scraping the lyrics page
    requires additional steps and must respect robots.txt and site policies.
    """

    def __init__(self, access_token: Optional[str] = None) -> None:
        self.access_token = access_token or os.getenv("GENIUS_ACCESS_TOKEN")
        self.base_url = "https://api.genius.com"
        self.session = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.logger = logging.getLogger(__name__)

        if self.access_token:
            self.session.headers.update({"Authorization": f"Bearer {self.access_token}"})

    def search_song(self, query: str, per_page: int = 5, page: int = 1) -> Optional[Dict[str, Any]]:
        if not self.access_token:
            self.logger.error("Missing GENIUS_ACCESS_TOKEN")
            return None
        params = {"q": query, "per_page": per_page, "page": page}
        try:
            resp = self.session.get(f"{self.base_url}/search", params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            self.logger.error(f"Genius search failed: {e}")
            return None

    def get_song(self, song_id: int) -> Optional[Dict[str, Any]]:
        if not self.access_token:
            self.logger.error("Missing GENIUS_ACCESS_TOKEN")
            return None
        try:
            resp = self.session.get(f"{self.base_url}/songs/{song_id}")
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            self.logger.error(f"Genius get_song failed: {e}")
            return None

    @staticmethod
    def best_match(hit_results: Dict[str, Any], track_name: str, artist_name: str) -> Optional[Dict[str, Any]]:
        """
        Return the best matching hit from Genius search results by simple heuristics.
        """
        hits = hit_results.get("response", {}).get("hits", [])
        if not hits:
            return None
        track_l = track_name.lower()
        artist_l = artist_name.lower()
        for hit in hits:
            full = hit.get("result", {}).get("full_title", "").lower()
            primary = hit.get("result", {}).get("primary_artist", {}).get("name", "").lower()
            if track_l in full and (artist_l in full or artist_l == primary):
                return hit.get("result")
        return hits[0].get("result") if hits else None
