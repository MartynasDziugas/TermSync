from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Enum, Float,
    ForeignKey, Integer, JSON, String, Text,
    UniqueConstraint, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass

class TermChangeType(str, enum.Enum):
    ADDED = "added"
    DELETED = "deleted"
    MODIFIED = "modified"
    PROMOTED = "promoted"


class DocumentStatus(str, enum.Enum):
    PENDING ="pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class MedDRALevel(str, enum.Enum):
    SOC = "soc"
    HLGT = "hlgt"
    PT = "pt"
    LLT = "llt"


class MedDRAVersion(Base):
    __tablename__ = "meddra_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_number: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    release_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="LT")
    imported_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    terms: Mapped[list[MedDRATerm]] = relationship("MedDRATerm", back_populates="version", cascade="all, delete-orphan")
    changes: Mapped[list[MedDRAChange]] = relationship("MedDRAChange", foreign_keys="MedDRAChange.new_version_id", back_populates="new_version")
    

class MedDRATerm(Base):
    __tablename__ = "meddra_terms"
    __table_args__ = (UniqueConstraint("version_id", "term_code", name="uq_term_version_code",),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    version_id: Mapped[int] = mapped_column(Integer, ForeignKey("meddra_versions.id", ondelete="CASCADE"), nullable=False)
    term_code: Mapped[str] = mapped_column(String(20), nullable=False)
    term_text: Mapped[str] = mapped_column(Text, nullable=False)
    level: Mapped[MedDRALevel] = mapped_column(Enum(MedDRALevel), nullable=False)
    parent_code: Mapped[str | None] = mapped_column(String(20), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)

    version: Mapped[MedDRAVersion] = relationship("MedDRAVersion", back_populates="terms")

class MedDRAChange(Base):
    __tablename__ = "meddra_changes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    old_version_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("meddra_versions.id", ondelete="SET NULL"), nullable=True)
    new_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("meddra_versions.id", ondelete="CASCADE"), nullable=False)
    term_code: Mapped[str] = mapped_column(String(20), nullable=False)
    change_type: Mapped[TermChangeType] = mapped_column(Enum(TermChangeType), nullable=False)
    old_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())

    new_version: Mapped[MedDRAVersion] = relationship("MedDRAVersion", foreign_keys=[new_version_id], back_populates="changes")

class DocumentTemplate(Base):
    __tablename__ = "document_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="LT")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    segments: Mapped[list[TemplateSegment]] = relationship("TemplateSegment", back_populates="template", cascade="all, delete-orphan")
    sync_jobs: Mapped[list[SyncJob]] = relationship("SyncJob", back_populates="template")

class TemplateSegment(Base):
    __tablename__ = "template_segments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    template_id: Mapped[int] = mapped_column(Integer, ForeignKey("document_templates.id", ondelete="CASCADE"), nullable=False)
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    translated_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    meddra_term_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("meddra_terms.id", ondelete="SET NULL"), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    
    template: Mapped[DocumentTemplate] = relationship("DocumentTemplate", back_populates="segments")
    meddra_term: Mapped[MedDRATerm | None] = relationship("MedDRATerm")

class SyncJob(Base):
    __tablename__ = "sync_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    template_id: Mapped[int] = mapped_column(Integer, ForeignKey("document_templates.id", ondelete="CASCADE"), nullable=False)
    meddra_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("meddra_versions.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(Enum(DocumentStatus), nullable=False, default=DocumentStatus.PENDING)
    segments_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    segments_updated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    template: Mapped[DocumentTemplate] = relationship("DocumentTemplate", back_populates="sync_jobs")
    meddra_version: Mapped[MedDRAVersion] = relationship("MedDRAVersion")

class ExperimentLog(Base):
    __tablename__ = "experiment_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    hyperparameters: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    dataset_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())