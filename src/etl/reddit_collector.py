"""
Reddit collector using PRAW to fetch mood-related posts.
"""
from __future__ import annotations
import os
import logging
from datetime import datetime, timezone
from typing import Iterable, List, Dict, Any, Optional

import praw


class RedditCollector:
    def __init__(self,
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 user_agent: Optional[str] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", "mood-spotify-ml/0.1")
        if not (self.client_id and self.client_secret and self.user_agent):
            self.logger.warning("Missing Reddit credentials; API calls will fail.")
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
        )

    def search(self,
               subreddits: List[str],
               query: str,
               limit: int = 100) -> Iterable[Dict[str, Any]]:
        """
        Search posts in given subreddits using Reddit's search.
        """
        for sub in subreddits:
            try:
                self.logger.info(f"Searching r/{sub} for '{query}' (limit={limit})")
                subreddit = self.reddit.subreddit(sub)
                for submission in subreddit.search(query=query, limit=limit, sort="new"):
                    yield self._to_record(submission)
            except Exception as e:
                self.logger.error(f"Error searching r/{sub}: {e}")

    @staticmethod
    def _to_record(submission) -> Dict[str, Any]:
        dt = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat()
        return {
            "platform": "reddit",
            "subreddit": getattr(submission, "subreddit", None).display_name if getattr(submission, "subreddit", None) else None,
            "id": submission.id,
            "title": submission.title,
            "selftext": submission.selftext,
            "url": submission.url,
            "score": submission.score,
            "num_comments": submission.num_comments,
            "created_at": dt,
            "author": str(submission.author) if submission.author else None,
            "permalink": f"https://reddit.com{submission.permalink}",
        }
