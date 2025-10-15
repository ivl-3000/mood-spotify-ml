"""
Playlist metadata collector for Spotify data.
"""
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .spotify_client import SpotifyClient


class PlaylistCollector:
    """Collects and processes Spotify playlist metadata."""
    
    def __init__(self, client: Optional[SpotifyClient] = None):
        """
        Initialize playlist collector.
        
        Args:
            client: Spotify client instance (creates new if None)
        """
        self.client = client or SpotifyClient()
        self.logger = logging.getLogger(__name__)
        
    def collect_playlist_metadata(self, playlist_id: str) -> Optional[Dict[str, Any]]:
        """
        Collect comprehensive metadata for a playlist.
        
        Args:
            playlist_id: Spotify playlist ID
            
        Returns:
            Dictionary with playlist metadata or None if failed
        """
        self.logger.info(f"Collecting metadata for playlist: {playlist_id}")
        
        # Get playlist info
        playlist_data = self.client.get_playlist(playlist_id)
        if not playlist_data:
            self.logger.error(f"Failed to get playlist data for {playlist_id}")
            return None
            
        # Get all tracks in playlist
        tracks_data = self._get_all_playlist_tracks(playlist_id)
        if not tracks_data:
            self.logger.warning(f"No tracks found for playlist {playlist_id}")
            tracks_data = []
            
        # Extract and structure metadata
        metadata = {
            'playlist_id': playlist_id,
            'name': playlist_data.get('name'),
            'description': playlist_data.get('description'),
            'owner': {
                'id': playlist_data.get('owner', {}).get('id'),
                'display_name': playlist_data.get('owner', {}).get('display_name'),
                'followers': playlist_data.get('owner', {}).get('followers', {}).get('total', 0)
            },
            'followers': playlist_data.get('followers', {}).get('total', 0),
            'public': playlist_data.get('public', False),
            'collaborative': playlist_data.get('collaborative', False),
            'created_at': playlist_data.get('created_at'),
            'updated_at': playlist_data.get('snapshot_id'),  # Using snapshot_id as update indicator
            'tracks_count': len(tracks_data),
            'tracks': tracks_data,
            'collected_at': datetime.now().isoformat()
        }
        
        self.logger.info(f"Collected metadata for playlist '{metadata['name']}' with {metadata['tracks_count']} tracks")
        return metadata
    
    def _get_all_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """
        Get all tracks from a playlist (handles pagination).
        
        Args:
            playlist_id: Spotify playlist ID
            
        Returns:
            List of track data dictionaries
        """
        all_tracks = []
        offset = 0
        limit = 100
        
        while True:
            tracks_response = self.client.get_playlist_tracks(playlist_id, limit=limit, offset=offset)
            if not tracks_response:
                break
                
            items = tracks_response.get('items', [])
            if not items:
                break
                
            # Process each track item
            for item in items:
                track_data = self._extract_track_data(item)
                if track_data:
                    all_tracks.append(track_data)
                    
            # Check if we have more tracks
            if len(items) < limit:
                break
                
            offset += limit
            
        return all_tracks
    
    def _extract_track_data(self, track_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract relevant track data from playlist track item.
        
        Args:
            track_item: Track item from playlist tracks API
            
        Returns:
            Structured track data or None if invalid
        """
        track = track_item.get('track')
        if not track or track.get('type') != 'track':
            return None
            
        # Get audio features
        audio_features = None
        if track.get('id'):
            audio_features = self.client.get_track_audio_features(track['id'])
            
        # Extract artist information
        artists = []
        for artist in track.get('artists', []):
            artists.append({
                'id': artist.get('id'),
                'name': artist.get('name'),
                'genres': [],  # Would need separate API call to get genres
                'popularity': 0  # Would need separate API call to get popularity
            })
            
        track_data = {
            'track_id': track.get('id'),
            'name': track.get('name'),
            'duration_ms': track.get('duration_ms'),
            'popularity': track.get('popularity'),
            'explicit': track.get('explicit', False),
            'preview_url': track.get('preview_url'),
            'external_urls': track.get('external_urls', {}),
            'album': {
                'id': track.get('album', {}).get('id'),
                'name': track.get('album', {}).get('name'),
                'release_date': track.get('album', {}).get('release_date'),
                'album_type': track.get('album', {}).get('album_type'),
                'total_tracks': track.get('album', {}).get('total_tracks'),
                'images': track.get('album', {}).get('images', [])
            },
            'artists': artists,
            'audio_features': audio_features,
            'added_at': track_item.get('added_at'),
            'added_by': track_item.get('added_by', {}).get('id') if track_item.get('added_by') else None
        }
        
        return track_data
    
    def save_playlist_metadata(self, metadata: Dict[str, Any], output_dir: str = "data/raw") -> str:
        """
        Save playlist metadata to JSON file.
        
        Args:
            metadata: Playlist metadata dictionary
            output_dir: Output directory path
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        playlist_id = metadata['playlist_id']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"playlist_{playlist_id}_{timestamp}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Saved playlist metadata to {filepath}")
        return str(filepath)
    
    def collect_multiple_playlists(self, playlist_ids: List[str], output_dir: str = "data/raw") -> List[str]:
        """
        Collect metadata for multiple playlists.
        
        Args:
            playlist_ids: List of Spotify playlist IDs
            output_dir: Output directory for saved files
            
        Returns:
            List of file paths where metadata was saved
        """
        saved_files = []
        
        for playlist_id in playlist_ids:
            try:
                metadata = self.collect_playlist_metadata(playlist_id)
                if metadata:
                    filepath = self.save_playlist_metadata(metadata, output_dir)
                    saved_files.append(filepath)
                else:
                    self.logger.error(f"Failed to collect metadata for playlist {playlist_id}")
                    
            except Exception as e:
                self.logger.error(f"Error collecting playlist {playlist_id}: {e}")
                
        return saved_files
