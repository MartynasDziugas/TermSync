from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from config import config
from src.database.models import Base

engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engnine, autocommit=False, autoflush=False)

def init-DB() -> None:
    Base.metadata.create_all(bind=engine)

def get_session() -> Session:
    return SessionLocal()

    