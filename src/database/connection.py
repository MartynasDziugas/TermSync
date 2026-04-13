"""SQLite + SQLAlchemy: sesijos ir lentelių kūrimas."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import config
from src.database.models import Base

_engine = create_engine(
    config.DATABASE_URL,
    echo=False,
    future=True,
)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    from config import BASE_DIR

    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=_engine)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
