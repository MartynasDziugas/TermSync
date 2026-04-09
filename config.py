import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "termsync-dev-key-2026")
    DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{BASE_DIR}/data/termsync.db")
    UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"
    RAW_DATA_FOLDER = BASE_DIR / "data" / "raw"
    EXPERIMENTS_FOLDER = BASE_DIR / "experiments"
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    BERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    PHRASE_SIMILARITY_THRESHOLD = 0.85
    TERM_SIMILARITY_THRESHOLD = 0.80
    ALLOWED_EXTENSIONS = {"tmx", "tbx", "docx", "asc", "txt", "csv"}
    TEMPLATE_SOURCES = ["EMA_SmPC", "EMA_PIL", "FDA_PI", "ICH_E6", "custom"]

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = DevelopmentConfig()    