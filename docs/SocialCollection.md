# Social Data Collection (Mood-Related)

## Sources
- Reddit via PRAW (API credentials required)
- Twitter (X) via snscrape (no API credentials required, subject to site changes)

## Setup

### Reddit Credentials
Create an app at `https://www.reddit.com/prefs/apps` and note:
- Client ID
- Client Secret
- User Agent (string that identifies your app)

Set environment variables or .env:
```bash
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=mood-spotify-ml/0.1
```

### Twitter via snscrape
No API key required, but scraping may break if site changes.

## Usage
Collect mood-related posts:
```bash
python scripts/collect_social.py --keywords happy sad chill calm --subreddits Music listentothis --reddit-limit 200 --twitter-limit 200 --outdir data/social
```

Outputs:
- `data/social/reddit_<timestamp>.jsonl`
- `data/social/twitter_<timestamp>.jsonl`

## Ethics and TOS
- Respect platform Terms of Service.
- Collect only public data; avoid PII.
- Use aggregates for modeling; do not redistribute raw content.
- Rate-limit responsibly; cache and deduplicate.
