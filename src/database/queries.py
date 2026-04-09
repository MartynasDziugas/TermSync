from sqlalchemy.orm import Session
from src.databse.models import MedDRAVersion, MedDRATerm, MedDRAChange, SyncJob, DocumentTemplate

def get_active_versions(session: Session) -> MedDRAVersion | None:
    return session.query(MedDRAVersion).filter_by(is_active=True).first()

def get_version_by_number(session: Session, version_number: str) -> MedDRAVersion | None:
    return session.query(MedDRATerm).filter_by(version_id=version_id).all()

def get_term_by_version(session: Session, version_id: int) -> list[MedDRATerm]:
    return session.query(MedDRATerm).filter_by(version-id=version_id).all()

def get_changes_between_version(
    session: Session, old_version_id: int, new_version_id: int
) -> list[MedDRAChange]:
    return (
        session.query(MedDRAChange)
        .filter_by(old_version_id=old_version_id, new_version_id=new_version_id)
        .all()
    )

def get_all_templates(session: Session) -> list[DocumentTemplate]:
    return session.query(DocumentTemplate).all()

def get_pending_sync_jobs(session: Session) -> list[SyncJob]:
    return session.query(SyncJob).filter_by(status="pending").all()
    