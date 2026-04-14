"""Vertėjo tikrinimo įkėlimų tęsinys: meta diske + sesijos be DB nuorodos."""

from __future__ import annotations

import json
import re
from pathlib import Path

from sqlalchemy import select

from config import config
from src.database.connection import get_session
from src.database.models import ReviewSession

SESSION_META_FILENAME = "session_meta.json"
UPLOAD_FOLDER_UID_RE = re.compile(r"^[a-f0-9]{12}$")
GLOSSARY_UPLOAD_DIR_UID_RE = re.compile(r"^[a-f0-9]{10}$")


def write_translator_session_meta(
    base_dir: Path,
    *,
    translator_src_display: str,
    translator_tgt_display: str,
    src_disk_name: str,
    tgt_disk_name: str,
    source_lang: str | None = None,
    target_lang: str | None = None,
) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, str] = {
        "translator_src_name": translator_src_display,
        "translator_tgt_name": translator_tgt_display,
        "src_disk_name": src_disk_name,
        "tgt_disk_name": tgt_disk_name,
    }
    if source_lang:
        payload["source_lang"] = source_lang
    if target_lang:
        payload["target_lang"] = target_lang
    (base_dir / SESSION_META_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _uid_from_src_rel(rel: str) -> str | None:
    parts = rel.replace("\\", "/").split("/")
    try:
        i = parts.index("translator_check")
        return parts[i + 1]
    except (ValueError, IndexError):
        return None


def _referenced_upload_uids() -> set[str]:
    uids: set[str] = set()
    with get_session() as session:
        for rs in session.scalars(select(ReviewSession)).all():
            u = _uid_from_src_rel(rs.translator_src_rel_path)
            if u:
                uids.add(u)
    return uids


def list_resumable_translator_uploads() -> list[dict[str, str]]:
    """Įkėlimų aplankai su session_meta.json, į kuriuos DB šiuo metu nenurodo."""
    root = config.UPLOAD_FOLDER / "translator_check"
    if not root.is_dir():
        return []
    ref = _referenced_upload_uids()
    out: list[dict[str, str]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        if not child.is_dir():
            continue
        uid = child.name
        if not UPLOAD_FOLDER_UID_RE.match(uid) or uid in ref:
            continue
        meta_path = child / SESSION_META_FILENAME
        if not meta_path.is_file():
            continue
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sn = data.get("src_disk_name")
        tn = data.get("tgt_disk_name")
        if not sn or not tn:
            continue
        if not (child / str(sn)).is_file() or not (child / str(tn)).is_file():
            continue
        out.append(
            {
                "uid": uid,
                "translator_src_name": str(data.get("translator_src_name", sn)),
                "translator_tgt_name": str(data.get("translator_tgt_name", tn)),
            }
        )
    return out


def _session_public_id_for_upload_uid(uid: str) -> str | None:
    needle = f"translator_check/{uid}/"
    with get_session() as session:
        rs = session.scalars(
            select(ReviewSession)
            .where(ReviewSession.translator_src_rel_path.contains(needle))
            .order_by(ReviewSession.created_at.desc())
        ).first()
        return rs.public_id if rs is not None else None


def list_glossary_disk_uploads() -> list[dict[str, str]]:
    """
    Į `glossary_csv/<10 hex>/` įrašyti CSV failai (Doc Upload arba kitas importas į DB).
    Rūšiuota pagal aplanko pakeitimo laiką (naujausi pirmi).
    """
    root = config.UPLOAD_FOLDER / "glossary_csv"
    if not root.is_dir():
        return []
    pairs: list[tuple[float, dict[str, str]]] = []
    for child in root.iterdir():
        if not child.is_dir() or not GLOSSARY_UPLOAD_DIR_UID_RE.match(child.name):
            continue
        for f in sorted(child.iterdir(), key=lambda p: p.name.lower()):
            if not f.is_file() or f.suffix.lower() != ".csv":
                continue
            try:
                mtime = f.stat().st_mtime_ns
            except OSError:
                mtime = 0.0
            rel = f.relative_to(config.UPLOAD_FOLDER)
            pairs.append(
                (
                    mtime,
                    {
                        "uid": child.name,
                        "filename": f.name,
                        "path_label": str(rel).replace("\\", "/"),
                    },
                )
            )
    pairs.sort(key=lambda t: t[0], reverse=True)
    return [t[1] for t in pairs]


def list_translator_disk_inventory() -> list[dict[str, str | None]]:
    """translator_check aplankai: su session_meta.json arba bent du .docx (rodoma istorijoje)."""
    root = config.UPLOAD_FOLDER / "translator_check"
    if not root.is_dir():
        return []
    out: list[dict[str, str | None]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        if not child.is_dir() or not UPLOAD_FOLDER_UID_RE.match(child.name):
            continue
        uid = child.name
        meta_path = child / SESSION_META_FILENAME
        if meta_path.is_file():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            sn = data.get("src_disk_name")
            tn = data.get("tgt_disk_name")
            if sn and tn and (child / str(sn)).is_file() and (child / str(tn)).is_file():
                out.append(
                    {
                        "uid": uid,
                        "translator_src_name": str(data.get("translator_src_name", sn)),
                        "translator_tgt_name": str(data.get("translator_tgt_name", tn)),
                        "session_public_id": _session_public_id_for_upload_uid(uid),
                    }
                )
                continue
        docxs = sorted(
            [p for p in child.iterdir() if p.is_file() and p.suffix.lower() == ".docx"],
            key=lambda p: p.name.lower(),
        )
        if len(docxs) >= 2:
            out.append(
                {
                    "uid": uid,
                    "translator_src_name": docxs[0].name,
                    "translator_tgt_name": docxs[1].name,
                    "session_public_id": _session_public_id_for_upload_uid(uid),
                }
            )
    return out


def resolve_paths_for_regenerate(uid: str) -> tuple[Path, Path, str, str] | None:
    """Grąžina (src_path, tgt_path, src_display, tgt_display) arba None."""
    if not UPLOAD_FOLDER_UID_RE.match(uid):
        return None
    base = config.UPLOAD_FOLDER / "translator_check" / uid
    meta_path = base / SESSION_META_FILENAME
    if not base.is_dir() or not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    sn = data.get("src_disk_name")
    tn = data.get("tgt_disk_name")
    if not sn or not tn:
        return None
    ts_path = base / str(sn)
    tt_path = base / str(tn)
    if not ts_path.is_file() or not tt_path.is_file():
        return None
    return (
        ts_path,
        tt_path,
        str(data.get("translator_src_name", sn))[:512],
        str(data.get("translator_tgt_name", tn))[:512],
    )
