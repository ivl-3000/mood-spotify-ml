from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Float, Boolean, DateTime, JSON, ForeignKey, Text


class Base(DeclarativeBase):
    pass


class Artist(Base):
    __tablename__ = "artists"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String)
    followers: Mapped[Optional[int]] = mapped_column(Integer)


class Album(Base):
    __tablename__ = "albums"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String)
    release_date: Mapped[Optional[String]] = mapped_column(String)
    label: Mapped[Optional[str]] = mapped_column(String)
    artist_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("artists.id"))


class Track(Base):
    __tablename__ = "tracks"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String)
    album_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("albums.id"))
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    popularity: Mapped[Optional[int]] = mapped_column(Integer)
    explicit: Mapped[Optional[bool]] = mapped_column(Boolean)


class AudioFeatures(Base):
    __tablename__ = "audio_features"
    track_id: Mapped[str] = mapped_column(String, ForeignKey("tracks.id"), primary_key=True)
    danceability: Mapped[Optional[Float]] = mapped_column(Float)
    energy: Mapped[Optional[Float]] = mapped_column(Float)
    valence: Mapped[Optional[Float]] = mapped_column(Float)
    tempo: Mapped[Optional[Float]] = mapped_column(Float)
    key: Mapped[Optional[int]] = mapped_column(Integer)
    mode: Mapped[Optional[int]] = mapped_column(Integer)
    loudness: Mapped[Optional[Float]] = mapped_column(Float)


class Playlist(Base):
    __tablename__ = "playlists"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String)
    description: Mapped[Optional[str]] = mapped_column(Text)
    owner_id: Mapped[Optional[str]] = mapped_column(String)
    owner_display_name: Mapped[Optional[str]] = mapped_column(String)
    followers: Mapped[Optional[int]] = mapped_column(Integer)
    public: Mapped[Optional[bool]] = mapped_column(Boolean)
    collaborative: Mapped[Optional[bool]] = mapped_column(Boolean)
    snapshot_id: Mapped[Optional[str]] = mapped_column(String)
    collected_at: Mapped[Optional[DateTime]] = mapped_column(DateTime)


class PlaylistTrack(Base):
    __tablename__ = "playlist_tracks"
    playlist_id: Mapped[str] = mapped_column(String, ForeignKey("playlists.id"), primary_key=True)
    track_id: Mapped[str] = mapped_column(String, ForeignKey("tracks.id"), primary_key=True)
    added_at: Mapped[Optional[DateTime]] = mapped_column(DateTime)
    added_by: Mapped[Optional[str]] = mapped_column(String)
    position: Mapped[Optional[int]] = mapped_column(Integer)


class Lyrics(Base):
    __tablename__ = "lyrics"
    track_id: Mapped[str] = mapped_column(String, ForeignKey("tracks.id"), primary_key=True)
    language: Mapped[Optional[str]] = mapped_column(String)
    text: Mapped[Optional[str]] = mapped_column(Text)
    source: Mapped[Optional[str]] = mapped_column(String)


class LyricsNLP(Base):
    __tablename__ = "lyrics_nlp"
    track_id: Mapped[str] = mapped_column(String, ForeignKey("tracks.id"), primary_key=True)
    sentiment_score: Mapped[Optional[Float]] = mapped_column(Float)
    emotion_probs: Mapped[Optional[dict]] = mapped_column(JSON)
    dominant_emotion: Mapped[Optional[str]] = mapped_column(String)


class SocialPost(Base):
    __tablename__ = "social_posts"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    platform: Mapped[Optional[str]] = mapped_column(String)
    subreddit: Mapped[Optional[str]] = mapped_column(String)
    title: Mapped[Optional[str]] = mapped_column(Text)
    content: Mapped[Optional[str]] = mapped_column(Text)
    url: Mapped[Optional[str]] = mapped_column(Text)
    author: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[Optional[DateTime]] = mapped_column(DateTime)
    score: Mapped[Optional[int]] = mapped_column(Integer)
    num_comments: Mapped[Optional[int]] = mapped_column(Integer)
    extra: Mapped[Optional[dict]] = mapped_column(JSON)
