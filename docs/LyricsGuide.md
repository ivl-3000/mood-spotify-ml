# Lyrics Extraction Guide

## Option A: Genius API (metadata + lyrics page URLs)

1. Create a Genius account and get an API Access Token:
   - Go to `https://genius.com/api-clients`
   - Create a client and copy the Access Token

2. Configure your environment:
```bash
# PowerShell
setx GENIUS_ACCESS_TOKEN "your_token_here"
# Or create/update .env
GENIUS_ACCESS_TOKEN=your_token_here
```

3. Use the playlist JSON from Spotify collector and run:
```bash
python scripts/collect_lyrics.py data/raw/playlist_<ID>_<TIMESTAMP>.json --output-dir data/lyrics
```

- Output: JSONL file with one line per track, including matched Genius `song_id`, `url`, and `full_title`.
- Note: The official Genius API does not return full lyrics text. If you plan to fetch lyrics content, respect robots.txt and site terms.

## Option B: Kaggle Lyrics Datasets (full text)

Use open datasets that permit research use. Examples (search on Kaggle):
- "Lyrics datasets" with track/artist fields
- "Genius lyrics" exports (check license)

Download and place under `data/raw/lyrics/` and map to Spotify tracks by `(track_name, artist_name)` with fuzzy matching or ISRC if available.

## Mapping Strategy
- Use `track_name + primary_artist` search query to find the best Genius hit.
- Maintain a cache/dictionary of matched `track_id -> genius_song_id/url` to avoid repeated searches.

## Next Steps
- Run NLP (sentiment/emotion) on available lyrics.
- Validate coverage and add fallbacks (lexicon-based) when lyrics are missing.
