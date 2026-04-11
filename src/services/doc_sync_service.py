from sqlalchemy.orm import Session
from src.database.models import DocumentTemplate, TemplateSegment, SyncJob, DocumentStatus
from src.models.bert_model import BertModel

bert = BertModel()


def create_sync_job(session: Session, old_template_id: int, new_template_id: int) -> SyncJob:
    try:
        job = SyncJob(
            template_id=new_template_id,
            meddra_version_id=old_template_id,
            status=DocumentStatus.PENDING,
        )
        session.add(job)
        session.commit()
        return job
    except Exception as e:
        session.rollback()
        raise RuntimeError(f"Klaida kuriant sinchronizavimo užduotį: {e}")


def run_sync_job(session: Session, job_id: int) -> SyncJob:
    try:
        job = session.get(SyncJob, job_id)
        if not job:
            raise ValueError(f"Užduotis {job_id} nerasta.")

        job.status = DocumentStatus.IN_PROGRESS
        session.commit()

        old_segments = (
            session.query(TemplateSegment)
            .filter_by(template_id=job.meddra_version_id)
            .order_by(TemplateSegment.segment_index)
            .all()
        )
        new_segments = (
            session.query(TemplateSegment)
            .filter_by(template_id=job.template_id)
            .order_by(TemplateSegment.segment_index)
            .all()
        )

        job.segments_total = len(new_segments)
        updated = 0

        for i, new_seg in enumerate(new_segments):
            if i < len(old_segments):
                score = bert.similarity(old_segments[i].source_text, new_seg.source_text)
                if score < 0.85:
                    new_seg.translated_text = f"[PASIKEITĖ] {new_seg.source_text}"
                    updated += 1

        job.segments_updated = updated
        job.status = DocumentStatus.COMPLETED
        session.commit()
        return job
    except Exception as e:
        session.rollback()
        if job:
            job.status = DocumentStatus.FAILED
            job.error_message = str(e)
            session.commit()
        raise RuntimeError(f"Klaida vykdant sinchronizavimą: {e}")