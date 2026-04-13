"""Pavojingos duomenų šalinimo operacijos (DB + failai diskelyje)."""

from __future__ import annotations

import shutil
from pathlib import Path

from sqlalchemy import select

from config import BASE_DIR, config
from src.database.connection import get_session
from src.database.models import ReviewSession, TrainingRun


def _safe_project_relative_path(rel: str) -> Path:
    rel_norm = rel.strip().lstrip("/")
    if ".." in Path(rel_norm).parts:
        raise ValueError("Netinkamas santykinis kelias.")
    p = (BASE_DIR / rel_norm).resolve()
    base = BASE_DIR.resolve()
    if not p.is_relative_to(base):
        raise ValueError("Kelias už projekto ribų.")
    return p


def _artifact_job_dir(artifact_path: str) -> Path | None:
    """Grąžina `.../standard_runs/<job_id>` katalogą, jei artefaktas ten."""
    try:
        raw = Path(artifact_path)
        ap = raw.resolve() if raw.is_absolute() else (BASE_DIR / raw).resolve()
    except OSError:
        return None
    root = config.STANDARD_RUNS_FOLDER.resolve()
    if not ap.is_relative_to(root):
        return None
    if ap.name == "bundle.joblib" and ap.parent.is_dir():
        return ap.parent
    if ap.is_dir():
        return ap
    return None


def purge_training_data() -> dict[str, int]:
    """
    Trinami visi standarto mokymai (DB + segmentai), kiekvieno job artefaktų aplankas,
    įkelti standarto .docx (`uploads/standard_train`) ir `LATEST` rodyklė.
    """
    dirs = 0
    with get_session() as session:
        runs = list(session.scalars(select(TrainingRun)).all())
        for tr in runs:
            jd = _artifact_job_dir(tr.artifact_path)
            if jd is not None and jd.is_dir():
                shutil.rmtree(jd, ignore_errors=True)
                dirs += 1
            session.delete(tr)

    n_runs = len(runs)

    std_up = config.UPLOAD_FOLDER / "standard_train"
    if std_up.is_dir():
        shutil.rmtree(std_up, ignore_errors=True)
        std_up.mkdir(parents=True, exist_ok=True)

    if config.LATEST_STANDARD_RUN_FILE.is_file():
        config.LATEST_STANDARD_RUN_FILE.unlink(missing_ok=True)

    return {"training_runs": n_runs, "artifact_job_dirs": dirs}


def purge_review_sessions(*, delete_files: bool) -> dict[str, int]:
    """Ištrina visas vertėjo peržiūros sesijas DB; pasirinktinai — susietus .docx diskelyje."""
    removed_rows = 0
    removed_files = 0
    with get_session() as session:
        rows = list(session.scalars(select(ReviewSession)).all())
        for rs in rows:
            if delete_files:
                for rel in (rs.translator_src_rel_path, rs.translator_tgt_rel_path):
                    try:
                        p = _safe_project_relative_path(rel)
                        if p.is_file():
                            p.unlink(missing_ok=True)
                            removed_files += 1
                    except (OSError, ValueError):
                        pass
            session.delete(rs)
            removed_rows += 1
    return {"sessions": removed_rows, "files": removed_files}


def purge_translator_check_uploads() -> dict[str, int]:
    """Išvalo `uploads/translator_check` (visi sesijų įkėlimai), DB nekeičia."""
    root = config.UPLOAD_FOLDER / "translator_check"
    if not root.is_dir():
        return {"entries": 0}
    n = 0
    for child in list(root.iterdir()):
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            elif child.is_file():
                child.unlink(missing_ok=True)
            n += 1
        except OSError:
            pass
    return {"entries": n}
