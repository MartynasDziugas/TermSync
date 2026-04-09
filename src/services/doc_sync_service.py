from sqlalchemy.orm import Session
from src.database.models import DocumentTemplate, TemplateSegment, SyncJob, DocumentStatus
from src.database.queries import get_active_version
from src.models.bert_model import BertModel

bert = BertModel()

def create_sync_job(session: Session, template_id: int) -> SyncJob:
    try:
        version = get_active_version(session)
        if not version:
            raise ValueError("Nepavyko rasti aktyvios versijos.")

        job = SyncJob(
            template_id=template_id,
            meddra_version_id=version.id,
            status=DocumentStatus.PENDING,
        )
        session.add(job)
        session.commit()
        return job
    except Exception as e:
        session.rollback()
        raise RuntimeError(f"Klaida kuriant sinchronizavimo uzduoti: {e}")
    
def run_sync_job(session: Session, job_id: int) -> SyncJob:
    try:
        job = session.get(SyncJob, job_id)
        if not job:
            raise ValueError(f"Uzduotis {job_id} nerasta.")

        job.status = DocumentStatus.IN_PROGRESS
        session.commit()

        segments = (
            session.query(TemplateSegment)
            .filter_by(template_id=job.template_id)
            .all()
        )

        job.segments_total = len(segments)
        updated = 0

        for segment in segments:
            if segment.meddra_term and segment.source_text:
                score = bert.similarity(
                    segment.source_text,
                    segment.meddra_term.term_text
                )
                if score < 0.80:
                    segment.translated_text = segment.meddra_term.term_text
                    updated += 1

        job.segment_updated = updated
        job.status = DocumentStatus.COMPLETED
        session.commit()
        return job
    except Exception as e:
        session.rollback()
        if job:
            job.status = DocumentStatus.FAILED
            job.error_message = str(e)
            session.commit()
        raise RuntimeError(f"Klaida vykdant sinchronizavima: {e}")
        