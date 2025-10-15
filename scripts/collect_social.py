#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from etl.reddit_collector import RedditCollector
from etl.twitter_collector import TwitterCollector


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('social_collection.log')]
    )


def write_jsonl(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Collect mood-related social posts (Reddit/Twitter)")
    parser.add_argument("--keywords", nargs="+", default=["happy", "sad", "chill", "angry", "calm"], help="Mood keywords")
    parser.add_argument("--subreddits", nargs="+", default=["Music", "listentothis"], help="Subreddits to search")
    parser.add_argument("--reddit-limit", type=int, default=200)
    parser.add_argument("--twitter-limit", type=int, default=200)
    parser.add_argument("--outdir", default="data/social", help="Output directory")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    setup_logging(args.log_level)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Reddit
    reddit = RedditCollector()
    records_reddit = []
    for kw in args.keywords:
        q = f"title:{kw} OR selftext:{kw}"
        for r in reddit.search(args.subreddits, q, limit=args.reddit_limit):
            records_reddit.append(r)
    out_reddit = Path(args.outdir) / f"reddit_{timestamp}.jsonl"
    write_jsonl(records_reddit, out_reddit)
    logging.getLogger(__name__).info(f"Saved Reddit posts: {out_reddit} ({len(records_reddit)})")

    # Twitter
    twitter = TwitterCollector()
    query = "(" + " OR ".join(args.keywords) + ") lang:en"
    records_twitter = list(twitter.search(query, limit=args.twitter_limit))
    out_twitter = Path(args.outdir) / f"twitter_{timestamp}.jsonl"
    write_jsonl(records_twitter, out_twitter)
    logging.getLogger(__name__).info(f"Saved Twitter posts: {out_twitter} ({len(records_twitter)})")


if __name__ == "__main__":
    main()
