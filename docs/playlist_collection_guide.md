# Playlist Collection Guide

## Overview
This guide explains how to collect Spotify playlist metadata using the ETL pipeline.

## Setup

### 1. Get Spotify API Credentials
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Note your Client ID and Client Secret

### 2. Configure Environment
```bash
# Copy the example config
cp config/env.example .env

# Edit .env with your credentials
SPOTIFY_CLIENT_ID=your_actual_client_id
SPOTIFY_CLIENT_SECRET=your_actual_client_secret
```

### 3. Activate Virtual Environment
```powershell
. .\.venv\Scripts\Activate.ps1
```

## Usage

### Collect Single Playlist
```bash
python scripts/collect_playlists.py 37i9dQZF1DXcBWIGoYBM5M
```

### Collect Multiple Playlists
```bash
python scripts/collect_playlists.py 37i9dQZF1DXcBWIGoYBM5M 37i9dQZF1DX0XUsuxWHRQdR 37i9dQZF1DX4Wsb4d7NKfP
```

### With Custom Output Directory
```bash
python scripts/collect_playlists.py --output-dir data/playlists 37i9dQZF1DXcBWIGoYBM5M
```

## Output Format

Each playlist generates a JSON file with:
- Playlist metadata (name, description, owner, followers)
- Track information (name, artists, album, audio features)
- Collection timestamp

Example structure:
```json
{
  "playlist_id": "37i9dQZF1DXcBWIGoYBM5M",
  "name": "Today's Top Hits",
  "description": "The most played songs right now",
  "owner": {
    "id": "spotify",
    "display_name": "Spotify",
    "followers": 0
  },
  "followers": 1234567,
  "tracks_count": 50,
  "tracks": [
    {
      "track_id": "4iV5W9uYEdYUVa79Axb7Rh",
      "name": "Song Name",
      "artists": [{"id": "artist_id", "name": "Artist Name"}],
      "album": {"id": "album_id", "name": "Album Name"},
      "audio_features": {
        "danceability": 0.8,
        "energy": 0.7,
        "valence": 0.6
      }
    }
  ],
  "collected_at": "2024-10-14T10:30:00"
}
```

## Popular Playlist IDs for Testing

- Today's Top Hits: `37i9dQZF1DXcBWIGoYBM5M`
- RapCaviar: `37i9dQZF1DX0XUsuxWHRQdR`
- Rock Classics: `37i9dQZF1DX4Wsb4d7NKfP`
- Chill Hits: `37i9dQZF1DX4WYpdgoIcn6`
- Mood Booster: `37i9dQZF1DX3rxVfibe1L0`

## Rate Limits
- Spotify API allows 100 requests per 100 seconds
- The client automatically handles rate limiting with retries
- Large playlists may take several minutes to collect

## Troubleshooting

### Authentication Errors
- Verify your Client ID and Secret are correct
- Ensure your Spotify app is not in development mode (if needed)

### Rate Limiting
- The script automatically handles rate limits
- For large collections, consider running during off-peak hours

### Missing Audio Features
- Some tracks may not have audio features available
- This is normal and handled gracefully by the collector
