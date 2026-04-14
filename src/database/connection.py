"""SQLite + SQLAlchemy: sesijos ir lentelių kūrimas."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
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
    _ensure_review_session_glossary_columns()


def _ensure_review_session_glossary_columns() -> None:
    """SQLite: prideda stulpelius senoms DB be Alembic."""
    if not str(config.DATABASE_URL).startswith("sqlite"):
        return
    with _engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(review_sessions)")).fetchall()
        col_names = {r[1] for r in rows}
        if "glossary_upload_filename" not in col_names:
            conn.execute(
                text(
                    "ALTER TABLE review_sessions ADD COLUMN glossary_upload_filename VARCHAR(512)"
                )
            )
        if "glossary_from_db" not in col_names:
            conn.execute(
                text("ALTER TABLE review_sessions ADD COLUMN glossary_from_db BOOLEAN DEFAULT 0")
            )


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
