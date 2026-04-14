"""Pavojingos duomenų šalinimo operacijos (DB + failai diskelyje)."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Literal

from sqlalchemy import select

from config import BASE_DIR, config
from src.database.connection import get_session
from src.database.models import ReviewSession, TrainingRun

_TRANSLATOR_CHECK_UID_RE = re.compile(r"^[a-f0-9]{12}$")


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


def delete_translator_check_upload(upload_uid: str) -> dict[str, int]:
    """
    Pašalina vieną `translator_check/<upload_uid>` aplanką.
    Trinamos ir DB `ReviewSession` eilutės, kurių keliai rodo į šį aplanką (kad neliktų nebegaliojančių nuorodų).
    """
    root = (config.UPLOAD_FOLDER / "translator_check" / upload_uid).resolve()
    base = (config.UPLOAD_FOLDER / "translator_check").resolve()
    if not root.is_relative_to(base) or root == base:
        raise ValueError("Netinkamas įkėlimo kelias.")
    needle = f"translator_check/{upload_uid}/"
    removed_sessions = 0
    with get_session() as session:
        rows = list(
            session.scalars(
                select(ReviewSession).where(ReviewSession.translator_src_rel_path.contains(needle))
            ).all()
        )
        for rs in rows:
            session.delete(rs)
            removed_sessions += 1
    if root.is_dir():
        shutil.rmtree(root, ignore_errors=True)
    return {"sessions": removed_sessions}


def purge_all_glossary_csv_disk() -> dict[str, int]:
    """Ištrina visą turinį po `uploads/glossary_csv/` (visi poaplankiai ir failai šaknyje)."""
    root = config.UPLOAD_FOLDER / "glossary_csv"
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


def purge_glossary_disk_files_named(secure_basename: str) -> None:
    """Ištrina visas `glossary_csv/*/<secure_basename>` kopijas; tuščius aplankus pašalina."""
    name = (secure_basename or "").strip()
    if not name or ".." in name or "/" in name or "\\" in name:
        return
    root = config.UPLOAD_FOLDER / "glossary_csv"
    if not root.is_dir():
        return
    for child in list(root.iterdir()):
        if not child.is_dir():
            continue
        fp = child / name
        if fp.is_file():
            try:
                fp.unlink()
            except OSError:
                pass
        try:
            if child.is_dir() and not any(child.iterdir()):
                child.rmdir()
        except OSError:
            pass


def purge_translator_uploads_having_docx_basenames(src_basename: str, tgt_basename: str) -> None:
    """Pašalina visus `translator_check/<uid>` įkėlimus, kuriuose yra .docx su vienu iš šių vardų."""
    names = {src_basename.strip(), tgt_basename.strip()}
    names.discard("")
    if not names:
        return
    base = config.UPLOAD_FOLDER / "translator_check"
    if not base.is_dir():
        return
    to_remove: list[str] = []
    for child in list(base.iterdir()):
        if not child.is_dir() or not _TRANSLATOR_CHECK_UID_RE.match(child.name):
            continue
        for p in child.glob("*.docx"):
            if p.name in names:
                to_remove.append(child.name)
                break
    for uid in to_remove:
        try:
            delete_translator_check_upload(uid)
        except ValueError:
            pass


_STANDARD_TRAIN_JOB_ID_RE = re.compile(r"^[a-f0-9]{16}$")


def delete_standard_train_upload_job_completely(job_id: str) -> None:
    """
    Pašalina standarto įkėlimą: `uploads/standard_train/<job_id>`, `standard_runs/<job_id>`,
    DB `TrainingRun` (su segmentais), atminties job, `LATEST` žymeklį jei rodė į šį job.
    """
    if not _STANDARD_TRAIN_JOB_ID_RE.match(job_id):
        return
    from src.services.pair_train_job_store import remove_job

    remove_job(job_id)

    with get_session() as session:
        tr = session.scalars(select(TrainingRun).where(TrainingRun.job_id == job_id)).first()
        if tr is not None:
            try:
                ap = Path(tr.artifact_path).resolve()
                if ap.is_file():
                    parent = ap.parent
                else:
                    parent = ap
                if parent.is_dir() and parent.name == job_id:
                    shutil.rmtree(parent, ignore_errors=True)
            except (OSError, ValueError):
                pass
            session.delete(tr)

    std_up = config.UPLOAD_FOLDER / "standard_train" / job_id
    if std_up.is_dir():
        shutil.rmtree(std_up, ignore_errors=True)
    art = config.STANDARD_RUNS_FOLDER / job_id
    if art.is_dir():
        shutil.rmtree(art, ignore_errors=True)

    if config.LATEST_STANDARD_RUN_FILE.is_file():
        try:
            cur = config.LATEST_STANDARD_RUN_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            cur = ""
        if cur == job_id:
            config.LATEST_STANDARD_RUN_FILE.unlink(missing_ok=True)


def purge_standard_train_uploads_matching_meta_pair(display_src: str, display_tgt: str) -> None:
    """
    Pašalina senus `standard_train/<job_id>` įrašus, kurių `upload_meta.json` sutampa
    su naujai keliama pora (tie patys rodomi failų vardai).
    """
    src = (display_src or "").strip()[:512]
    tgt = (display_tgt or "").strip()[:512]
    if not src or not tgt:
        return
    root = config.UPLOAD_FOLDER / "standard_train"
    if not root.is_dir():
        return
    for d in list(root.iterdir()):
        if not d.is_dir() or not _STANDARD_TRAIN_JOB_ID_RE.match(d.name):
            continue
        meta_path = d / "upload_meta.json"
        if not meta_path.is_file():
            continue
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if not isinstance(raw, dict):
            continue
        if (
            str(raw.get("standard_src_name") or "") == src
            and str(raw.get("standard_tgt_name") or "") == tgt
        ):
            delete_standard_train_upload_job_completely(d.name)


def list_standard_train_upload_pairs() -> list[dict[str, str | bool]]:
    """
    Įkeltos standarto poros: `uploads/standard_train/<job_id>/`.
    Įtraukiami ir nepilni įkėlimai (tik source arba tik target diske).
    Rūšiuota pagal naujausią pirmą (mtime).
    Kiekvienas elementas: job_id, src_name, tgt_name, has_src, has_tgt.
    """
    root = config.UPLOAD_FOLDER / "standard_train"
    if not root.is_dir():
        return []
    pairs: list[tuple[float, dict[str, str | bool]]] = []
    for d in root.iterdir():
        if not d.is_dir() or not _STANDARD_TRAIN_JOB_ID_RE.match(d.name):
            continue
        left = d / "standard_src.docx"
        right = d / "standard_tgt.docx"
        has_src = left.is_file()
        has_tgt = right.is_file()
        if not has_src and not has_tgt:
            continue
        meta_path = d / "upload_meta.json"
        src_name = "standard_src.docx"
        tgt_name = "standard_tgt.docx"
        if meta_path.is_file():
            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    src_name = str(raw.get("standard_src_name") or src_name)[:512]
                    tgt_name = str(raw.get("standard_tgt_name") or tgt_name)[:512]
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        try:
            mtime = d.stat().st_mtime_ns
        except OSError:
            mtime = 0.0
        pairs.append(
            (
                mtime,
                {
                    "job_id": d.name,
                    "src_name": src_name,
                    "tgt_name": tgt_name,
                    "has_src": has_src,
                    "has_tgt": has_tgt,
                },
            )
        )
    pairs.sort(key=lambda t: t[0], reverse=True)
    return [t[1] for t in pairs]


StandardTrainDeleteResult = Literal["none", "file", "folder"]


def delete_standard_train_upload_file(job_id: str, role: str) -> StandardTrainDeleteResult:
    """
    Ištrina vieną `standard_src.docx` (role='src') arba `standard_tgt.docx` (role='tgt').
    Jei nelieka nė vieno .docx, pašalinamas visas aplankas.
    """
    if not _STANDARD_TRAIN_JOB_ID_RE.match(job_id) or role not in ("src", "tgt"):
        return "none"
    p = (config.UPLOAD_FOLDER / "standard_train" / job_id).resolve()
    base = (config.UPLOAD_FOLDER / "standard_train").resolve()
    if not p.is_relative_to(base) or p == base or not p.is_dir():
        return "none"
    fname = "standard_src.docx" if role == "src" else "standard_tgt.docx"
    fp = p / fname
    removed_file = False
    if fp.is_file():
        try:
            fp.unlink()
            removed_file = True
        except OSError:
            pass
    src_left = (p / "standard_src.docx").is_file()
    tgt_left = (p / "standard_tgt.docx").is_file()
    if not src_left and not tgt_left:
        shutil.rmtree(p, ignore_errors=True)
        return "folder"
    return "file" if removed_file else "none"


def delete_all_standard_train_uploads() -> list[str]:
    """Ištrina visus tinkamus `standard_train/<job_id>` aplankus. Grąžina pašalintų job_id sąrašą."""
    root = config.UPLOAD_FOLDER / "standard_train"
    if not root.is_dir():
        return []
    removed: list[str] = []
    for child in list(root.iterdir()):
        if not child.is_dir() or not _STANDARD_TRAIN_JOB_ID_RE.match(child.name):
            continue
        shutil.rmtree(child, ignore_errors=True)
        removed.append(child.name)
    return removed
