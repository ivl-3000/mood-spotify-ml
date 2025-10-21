#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy.orm import Session

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from db.engine import init_db, get_session_factory
from db.models import Playlist, PlaylistTrack, Track, Album, Artist, AudioFeatures, Lyrics, LyricsNLP, SocialPost


def setup_logging(level: str = "INFO"):
    logging.basicConfig(level=getattr(logging, level.upper()), format='%(asctime)s - %(levelname)s - %(message)s')


def load_playlist_json(path: Path, session: Session) -> None:
    data = json.loads(path.read_text(encoding='utf-8'))
    # Playlist
    pl = Playlist(
        id=data["playlist_id"],
        name=data.get("name"),
        description=data.get("description"),
        owner_id=((data.get("owner") or {}).get("id")),
        owner_display_name=((data.get("owner") or {}).get("display_name")),
        followers=(data.get("followers") or 0),
        public=data.get("public"),
        collaborative=data.get("collaborative"),
        snapshot_id=data.get("updated_at"),
        collected_at=datetime.fromisoformat(data.get("collected_at")) if data.get("collected_at") else None,
    )
    session.merge(pl)

    for t in data.get("tracks", []):
        # Artists
        for a in t.get("artists", []):
            session.merge(Artist(id=a.get("id"), name=a.get("name")))
        # Album
        album = t.get("album") or {}
        if album.get("id"):
            session.merge(Album(id=album["id"], name=album.get("name"), release_date=album.get("release_date")))
        # Track
        session.merge(Track(
            id=t.get("track_id"),
            name=t.get("name"),
            album_id=album.get("id"),
            duration_ms=t.get("duration_ms"),
            popularity=t.get("popularity"),
            explicit=t.get("explicit"),
        ))
        # Audio features
        af = t.get("audio_features")
        if af and t.get("track_id"):
            session.merge(AudioFeatures(
                track_id=t["track_id"],
                danceability=af.get("danceability"),
                energy=af.get("energy"),
                valence=af.get("valence"),
                tempo=af.get("tempo"),
                key=af.get("key"),
                mode=af.get("mode"),
                loudness=af.get("loudness"),
            ))
        # PlaylistTrack
        session.merge(PlaylistTrack(
            playlist_id=data["playlist_id"],
            track_id=t.get("track_id"),
            added_at=datetime.fromisoformat(t.get("added_at")) if t.get("added_at") else None,
            added_by=t.get("added_by"),
        ))


def load_lyrics_jsonl(path: Path, session: Session) -> None:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            # Store Genius metadata in Lyrics table (no full text here)
            genius = rec.get("genius") or {}
            session.merge(Lyrics(
                track_id=rec.get("track_id"),
                language=None,
                text=None,
                source=genius.get("genius_url"),
            ))


def load_social_jsonl(path: Path, session: Session) -> None:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            created = rec.get("created_at")
            dt = None
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                except Exception:
                    dt = None
            session.merge(SocialPost(
                id=str(rec.get("id")),
                platform=rec.get("platform"),
                subreddit=rec.get("subreddit"),
                title=rec.get("title"),
                content=rec.get("selftext") or rec.get("content"),
                url=rec.get("url"),
                author=rec.get("author") or rec.get("username"),
                created_at=dt,
                score=rec.get("score"),
                num_comments=rec.get("num_comments"),
                extra=rec,
            ))


def main():
    parser = argparse.ArgumentParser(description="Init DB and load raw JSON/JSONL data")
    parser.add_argument("--init", action="store_true", help="Create tables")
    parser.add_argument("--playlist-json", help="Path to playlist JSON")
    parser.add_argument("--lyrics-jsonl", help="Path to lyrics JSONL")
    parser.add_argument("--social-jsonl", help="Path to social JSONL")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.init:
        init_db()

    SessionFactory = get_session_factory()
    with SessionFactory() as session:
        if args.playlist_json:
            load_playlist_json(Path(args.playlist_json), session)
        if args.lyrics_jsonl:
            load_lyrics_jsonl(Path(args.lyrics_jsonl), session)
        if args.social_jsonl:
            load_social_jsonl(Path(args.social_jsonl), session)
        session.commit()


if __name__ == "__main__":
    main()
