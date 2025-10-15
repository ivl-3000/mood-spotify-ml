"""
Spotify Web API client for collecting playlist metadata.
"""
import os
import time
import logging
from typing import Dict, List, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class SpotifyClient:
    """Client for interacting with Spotify Web API."""
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        """
        Initialize Spotify client.
        
        Args:
            client_id: Spotify client ID (defaults to env var SPOTIFY_CLIENT_ID)
            client_secret: Spotify client secret (defaults to env var SPOTIFY_CLIENT_SECRET)
        """
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        self.base_url = "https://api.spotify.com/v1"
        self.access_token = None
        
        # Setup session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.logger = logging.getLogger(__name__)
        
    def authenticate(self) -> bool:
        """
        Authenticate with Spotify using Client Credentials flow.
        
        Returns:
            True if authentication successful, False otherwise
        """
        if not self.client_id or not self.client_secret:
            self.logger.error("Missing Spotify client credentials")
            return False
            
        auth_url = "https://accounts.spotify.com/api/token"
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        try:
            response = self.session.post(auth_url, data=auth_data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            self.session.headers.update({
                'Authorization': f'Bearer {self.access_token}'
            })
            
            self.logger.info("Successfully authenticated with Spotify")
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Authentication failed: {e}")
            return False
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make authenticated request to Spotify API.
        
        Args:
            endpoint: API endpoint (relative to base_url)
            params: Query parameters
            
        Returns:
            JSON response or None if failed
        """
        if not self.access_token:
            if not self.authenticate():
                return None
                
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(url, params=params)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 1))
                self.logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self._make_request(endpoint, params)
                
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    self.logger.error(f"Error details: {error_detail}")
                except:
                    self.logger.error(f"Response text: {e.response.text}")
            return None
    
    def get_playlist(self, playlist_id: str, market: str = "US") -> Optional[Dict]:
        """
        Get playlist metadata.
        
        Args:
            playlist_id: Spotify playlist ID (can be just ID or spotify:playlist:ID format)
            market: Market code (e.g., "US", "GB")
            
        Returns:
            Playlist data or None if failed
        """
        # Clean playlist ID (remove spotify:playlist: prefix if present)
        clean_id = playlist_id.replace("spotify:playlist:", "")
        params = {"market": market}
        return self._make_request(f"playlists/{clean_id}", params)
    
    def get_playlist_tracks(self, playlist_id: str, limit: int = 100, offset: int = 0, market: str = "US") -> Optional[Dict]:
        """
        Get tracks from a playlist.
        
        Args:
            playlist_id: Spotify playlist ID (can be just ID or spotify:playlist:ID format)
            limit: Number of tracks to retrieve (max 100)
            offset: Offset for pagination
            market: Market code (e.g., "US", "GB")
            
        Returns:
            Tracks data or None if failed
        """
        # Clean playlist ID (remove spotify:playlist: prefix if present)
        clean_id = playlist_id.replace("spotify:playlist:", "")
        params = {'limit': min(limit, 100), 'offset': offset, 'market': market}
        return self._make_request(f"playlists/{clean_id}/tracks", params)
    
    def get_track_audio_features(self, track_id: str) -> Optional[Dict]:
        """
        Get audio features for a track.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Audio features data or None if failed
        """
        return self._make_request(f"audio-features/{track_id}")
    
    def get_track(self, track_id: str) -> Optional[Dict]:
        """
        Get track metadata.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Track data or None if failed
        """
        return self._make_request(f"tracks/{track_id}")
    
    def get_artist(self, artist_id: str) -> Optional[Dict]:
        """
        Get artist metadata.
        
        Args:
            artist_id: Spotify artist ID
            
        Returns:
            Artist data or None if failed
        """
        return self._make_request(f"artists/{artist_id}")
    
    def get_album(self, album_id: str) -> Optional[Dict]:
        """
        Get album metadata.
        
        Args:
            album_id: Spotify album ID
            
        Returns:
            Album data or None if failed
        """
        return self._make_request(f"albums/{album_id}")
