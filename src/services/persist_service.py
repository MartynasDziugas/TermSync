"""Mokymo ir peržiūrų įrašymas į DB (SQLAlchemy ORM)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from sqlalchemy import func, select, update

from config import BASE_DIR, config
from src.database.connection import get_session
from src.database.models import (
    GlossaryBatch,
    GlossaryRow,
    ReviewSession,
    StandardSegment,
    TrainingRun,
)
from src.services.translator_review_service import ReviewRow


def _rel_to_project(p: Path) -> str:
    return str(p.resolve().relative_to(BASE_DIR))


def persist_training_run(
    *,
    job_id: str,
    standard_src_name: str,
    standard_tgt_name: str,
    aligned: list[tuple[str, str]],
    result: dict,
    artifact_path: str,
) -> None:
    with get_session() as session:
        session.execute(update(TrainingRun).values(is_active=False))
        run = TrainingRun(
            job_id=job_id,
            standard_src_name=standard_src_name[:512],
            standard_tgt_name=standard_tgt_name[:512],
            n_aligned_pairs=int(result["n_aligned_pairs"]),
            n_train_rows=int(result["n_train_rows"]),
            n_test_rows=int(result["n_test_rows"]),
            svm_test_accuracy=float(result["svm_test_accuracy"]),
            mlp_test_accuracy=float(result["mlp_test_accuracy"]),
            artifact_path=artifact_path,
            is_active=True,
        )
        session.add(run)
        session.flush()
        for idx, (src, tgt) in enumerate(aligned):
            session.add(
                StandardSegment(
                    training_run_id=run.id,
                    pair_index=idx,
                    source_text=src,
                    target_text=tgt,
                    is_synthetic_negative=False,
                )
            )
    config.LATEST_STANDARD_RUN_FILE.write_text(job_id, encoding="utf-8")


def persist_review_session(
    *,
    public_id: str,
    training_run_job_id: str,
    translator_src_name: str,
    translator_tgt_name: str,
    translator_src_path: Path,
    translator_tgt_path: Path,
    rows: list[ReviewRow],
) -> None:
    rows_payload = [asdict(r) for r in rows]
    with get_session() as session:
        session.add(
            ReviewSession(
                public_id=public_id,
                training_run_job_id=training_run_job_id,
                translator_src_name=translator_src_name[:512],
                translator_tgt_name=translator_tgt_name[:512],
                translator_src_rel_path=_rel_to_project(translator_src_path),
                translator_tgt_rel_path=_rel_to_project(translator_tgt_path),
                rows_json=json.dumps(rows_payload, ensure_ascii=False),
            )
        )


def get_active_training_run() -> TrainingRun | None:
    with get_session() as session:
        tr = session.scalars(
            select(TrainingRun)
            .where(TrainingRun.is_active.is_(True))
            .order_by(TrainingRun.created_at.desc())
        ).first()
        if tr is not None:
            session.expunge(tr)
        return tr


def list_training_history() -> list[TrainingRun]:
    with get_session() as session:
        rows = list(
            session.scalars(select(TrainingRun).order_by(TrainingRun.created_at.desc())).all()
        )
        for r in rows:
            session.expunge(r)
        return rows


def list_review_history() -> list[ReviewSession]:
    with get_session() as session:
        rows = list(
            session.scalars(
                select(ReviewSession).order_by(ReviewSession.created_at.desc())
            ).all()
        )
        for r in rows:
            session.expunge(r)
        return rows


def get_review_session(public_id: str) -> ReviewSession | None:
    with get_session() as session:
        r = session.scalars(
            select(ReviewSession).where(ReviewSession.public_id == public_id)
        ).first()
        if r is not None:
            session.expunge(r)
        return r


def delete_review_session(public_id: str, *, delete_files: bool = False) -> bool:
    """Pašalina vieną peržiūros sesiją. Pagal nutylėjimą tik DB; failai lieka diske."""
    from src.services.data_maintenance_service import _safe_project_relative_path

    with get_session() as session:
        r = session.scalars(
            select(ReviewSession).where(ReviewSession.public_id == public_id)
        ).first()
        if r is None:
            return False
        if delete_files:
            for rel in (r.translator_src_rel_path, r.translator_tgt_rel_path):
                try:
                    p = _safe_project_relative_path(rel)
                    if p.is_file():
                        p.unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass
        session.delete(r)
    return True


def review_rows_from_session(row: ReviewSession) -> list[ReviewRow]:
    data = json.loads(row.rows_json)
    out: list[ReviewRow] = []
    for item in data:
        item.setdefault("glossary_html", "")
        out.append(ReviewRow(**item))
    return out


def count_all_glossary_rows() -> int:
    with get_session() as session:
        n = session.scalar(select(func.count()).select_from(GlossaryRow))
    return int(n or 0)


def load_glossary_entries_for_review(
    max_rows: int | None = None,
) -> list[tuple[str, str, str | None]]:
    """(source, target, note), DB rikiuota pagal ilgiausią source (pirmenybė ilgesniems terminams)."""
    limit = max_rows if max_rows is not None else int(config.GLOSSARY_REVIEW_MAX_ROWS)
    with get_session() as session:
        rows = session.execute(
            select(GlossaryRow.source_text, GlossaryRow.target_text, GlossaryRow.note)
            .order_by(func.length(GlossaryRow.source_text).desc())
            .limit(limit)
        ).all()
    out: list[tuple[str, str, str | None]] = []
    for src, tgt, note in rows:
        s = (src or "").strip()
        t = (tgt or "").strip()
        if len(s) < int(config.GLOSSARY_REVIEW_MIN_SOURCE_LEN) or not t:
            continue
        n = (note or "").strip() or None
        out.append((s, t, n))
    return out


def persist_glossary_csv(
    *,
    filename: str,
    rows: list[tuple[str, str, str | None]],
) -> int:
    """Įrašo naują partiją ir eilutes. Grąžina batch.id."""
    with get_session() as session:
        batch = GlossaryBatch(
            filename=filename[:512],
            row_count=len(rows),
        )
        session.add(batch)
        session.flush()
        for idx, (src, tgt, note) in enumerate(rows):
            session.add(
                GlossaryRow(
                    batch_id=batch.id,
                    row_index=idx,
                    source_text=src,
                    target_text=tgt,
                    note=note,
                )
            )
        return int(batch.id)


def list_glossary_batches() -> list[GlossaryBatch]:
    with get_session() as session:
        rows = list(
            session.scalars(
                select(GlossaryBatch).order_by(GlossaryBatch.created_at.desc())
            ).all()
        )
        for r in rows:
            session.expunge(r)
        return rows


def get_glossary_batch(batch_id: int) -> GlossaryBatch | None:
    with get_session() as session:
        b = session.scalars(
            select(GlossaryBatch).where(GlossaryBatch.id == batch_id)
        ).first()
        if b is not None:
            session.expunge(b)
        return b


def list_glossary_rows(batch_id: int, limit: int = 500) -> list[GlossaryRow]:
    with get_session() as session:
        rows = list(
            session.scalars(
                select(GlossaryRow)
                .where(GlossaryRow.batch_id == batch_id)
                .order_by(GlossaryRow.row_index)
                .limit(limit)
            ).all()
        )
        for r in rows:
            session.expunge(r)
        return rows


def count_glossary_rows(batch_id: int) -> int:
    with get_session() as session:
        n = session.scalar(
            select(func.count()).select_from(GlossaryRow).where(GlossaryRow.batch_id == batch_id)
        )
        return int(n or 0)
