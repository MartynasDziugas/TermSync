"""SQLAlchemy ORM: mokymo istorija, standarto segmentai, vertėjo peržiūros sesijos."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


if TYPE_CHECKING:
    pass


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
    )
    standard_src_name: Mapped[str] = mapped_column(String(512))
    standard_tgt_name: Mapped[str] = mapped_column(String(512))
    n_aligned_pairs: Mapped[int] = mapped_column(Integer)
    n_train_rows: Mapped[int] = mapped_column(Integer)
    n_test_rows: Mapped[int] = mapped_column(Integer)
    svm_test_accuracy: Mapped[float] = mapped_column()
    mlp_test_accuracy: Mapped[float] = mapped_column()
    artifact_path: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    # Vertėjo peržiūroje naudojamas klasifikatorius (tas pats bundle: SVM + MLP galvos).
    active_model_type: Mapped[str] = mapped_column(String(8), default="svm")

    segments: Mapped[list["StandardSegment"]] = relationship(
        back_populates="training_run",
        cascade="all, delete-orphan",
    )


class StandardSegment(Base):
    __tablename__ = "standard_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_run_id: Mapped[int] = mapped_column(
        ForeignKey("training_runs.id", ondelete="CASCADE"),
        index=True,
    )
    pair_index: Mapped[int] = mapped_column(Integer)
    source_text: Mapped[str] = mapped_column(Text)
    target_text: Mapped[str] = mapped_column(Text)
    is_synthetic_negative: Mapped[bool] = mapped_column(Boolean, default=False)

    training_run: Mapped["TrainingRun"] = relationship(back_populates="segments")


class ReviewSession(Base):
    __tablename__ = "review_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
    )
    training_run_job_id: Mapped[str] = mapped_column(String(32), index=True)
    translator_src_name: Mapped[str] = mapped_column(String(512))
    translator_tgt_name: Mapped[str] = mapped_column(String(512))
    translator_src_rel_path: Mapped[str] = mapped_column(Text)
    translator_tgt_rel_path: Mapped[str] = mapped_column(Text)
    rows_json: Mapped[str] = mapped_column(Text)
    glossary_upload_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    glossary_from_db: Mapped[bool] = mapped_column(Boolean, default=False)


class GlossaryBatch(Base):
    """Vienas CSV importas (terminų lentelė)."""

    __tablename__ = "glossary_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
    )
    filename: Mapped[str] = mapped_column(String(512))
    row_count: Mapped[int] = mapped_column(Integer, default=0)

    rows: Mapped[list["GlossaryRow"]] = relationship(
        back_populates="batch",
        cascade="all, delete-orphan",
    )


class GlossaryRow(Base):
    """Viena CSV eilutė: source / target (+ neprivaloma pastaba)."""

    __tablename__ = "glossary_rows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("glossary_batches.id", ondelete="CASCADE"),
        index=True,
    )
    row_index: Mapped[int] = mapped_column(Integer)
    source_text: Mapped[str] = mapped_column(Text)
    target_text: Mapped[str] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    batch: Mapped["GlossaryBatch"] = relationship(back_populates="rows")
