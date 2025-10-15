"""
Twitter collector using snscrape to fetch tweets by query.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Iterable, Dict, Any

import snscrape.modules.twitter as sntwitter


class TwitterCollector:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def search(self, query: string, limit: int = 200) -> Iterable[Dict[str, Any]]:
        """
        Search recent tweets using snscrape query syntax.
        Example query: "(happy OR sad OR chill) lang:en"
        """
        count = 0
        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            yield self._to_record(tweet)
            count += 1
            if count >= limit:
                break

    @staticmethod
    def _to_record(tweet) -> Dict[str, Any]:
        created = tweet.date.astimezone(timezone.utc).isoformat() if tweet.date else None
        return {
            "platform": "twitter",
            "id": str(tweet.id),
            "content": tweet.rawContent,
            "username": tweet.user.username if tweet.user else None,
            "url": tweet.url,
            "created_at": created,
            "reply_count": getattr(tweet, "replyCount", None),
            "retweet_count": getattr(tweet, "retweetCount", None),
            "like_count": getattr(tweet, "likeCount", None),
            "lang": getattr(tweet, "lang", None),
            "hashtags": [h for h in (getattr(tweet, "hashtags", []) or [])],
        }
