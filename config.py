import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

class Config:
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get("SECRET_KEY", "termsync-dev-key-2026")
    DATABASE_URL = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{BASE_DIR / 'data' / 'termsync.db'}",
    )
    UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"
    STANDARD_RUNS_FOLDER = BASE_DIR / "data" / "standard_runs"
    LATEST_STANDARD_RUN_FILE = STANDARD_RUNS_FOLDER / "LATEST"
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    BERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    PHRASE_SIMILARITY_THRESHOLD = 0.85
    # Vertėjo target vs standarto target (kosinusas embeddingų), papildomas signalas prie SVM
    REVIEW_TGT_SIM_THRESHOLD = 0.78
    REVIEW_SRC_MIN_SIM = 0.52
    # --- Vertėjo tikrinimo porinimas (kalibravimas) ---
    # REVIEW_PAIR_ALIGN_MODE:
    #   "embedding" — kiekvienai pastraipai renkama artimiausia standarto pora visame rinkinyje (lankstu, gali klysti).
    #   "positional" — i-oji vertėjo pastraipa visada lyginama su i-ąja standarto pora (stipru, jei abu .docx
    #     sutampa pastraipų skaičiumi ir tvarka, kaip mokyme).
    # Aplinka: TERMSYNC_REVIEW_PAIR_ALIGN_MODE=positional
    _raw_align = os.environ.get("TERMSYNC_REVIEW_PAIR_ALIGN_MODE", "embedding").strip().lower()
    REVIEW_PAIR_ALIGN_MODE = _raw_align if _raw_align in ("embedding", "positional") else "embedding"
    # Jei positional ir vertėjo pastraipų daugiau nei porų: True = kartoti paskutinę porą, False = silpna eilutė
    REVIEW_POSITIONAL_PAD_LAST_PAIR = True
    # Poros rodymui (embedding režime — ir „spraga“ tarp 1-os ir 2-os EN vietos):
    REVIEW_PAIR_ACCEPT_MIN_SRC_COSINE = 0.50
    REVIEW_PAIR_ACCEPT_MIN_TGT_COSINE = 0.38
    REVIEW_PAIR_SRC_TOP1_TOP2_MARGIN = 0.05
    REVIEW_PAIR_STRONG_SRC_COSINE = 0.64
    # Standarto mokyme naudojamas greitas MLP (test metrikoms)
    QUICK_MLP_EPOCHS = 40
    # Embeddingų LRU kešas (ta pati eilutė pakartotiniuose paleidimuose)
    EMBEDDING_CACHE_MAX_ITEMS = 20_000
    # CSV žodynas vertėjo peržiūroje: max eilučių iš DB, min. source ilgis simboliais
    GLOSSARY_REVIEW_MAX_ROWS = 8000
    GLOSSARY_REVIEW_MIN_SOURCE_LEN = 2
    # Levenshteino atstumas (NFC + casefold tekstuose) šaltinio / target atitikčiai
    _gloss_ed = os.environ.get("TERMSYNC_GLOSSARY_MAX_EDIT_DISTANCE", "3").strip()
    try:
        GLOSSARY_MAX_EDIT_DISTANCE = max(0, min(5, int(_gloss_ed)))
    except ValueError:
        GLOSSARY_MAX_EDIT_DISTANCE = 3

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = DevelopmentConfig()    