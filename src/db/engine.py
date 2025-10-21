from __future__ import annotations
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base


def get_engine():
    database_url = os.getenv("DATABASE_URL", "sqlite:///data/spotify.db")
    # For SQLite, ensure directory exists
    if database_url.startswith("sqlite"):
        import pathlib
        pathlib.Path("data").mkdir(parents=True, exist_ok=True)
    return create_engine(database_url, future=True)


def get_session_factory():
    engine = get_engine()
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
