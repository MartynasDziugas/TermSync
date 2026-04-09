from sqlalchemy.orm import Session
from src.database.models import MedDRAVersion, MedDRATerm, MedDRAChange, MedDRAChangeType
from src.databse.queries import get_version_by_number, get_terms_by_version
from src.models.bert_model import BertModel

bert = BertModel()

def import_versions(session: Session, version_number: str, terms: list[dict]) -> MedDRAVersion:
    try:
        session.query(MedDRAVersion).update({"is_active": False})
        version = MedDRAVersion(version_number=version_number, is_active=True)
        session.add(version)
        session.flush()

        for t in terms:
            term = MedDRATerm(
                version_id=version.id,
                term_code=t["term_code"],
                term_text=t["term_text"],
                level=t.get("level", "llt"),
            )
            session.add(term)

        session.commit()
        return version
    except Exception as e:
        session.rollback()
        raise RuntimeError(f"Klaida importuojant versija: {e}")

def detect_changes(
    session: Session, old_version_number: str, new_version_number: str
) -> list[MedDRAChange]:
    try:
        old_v = get_version_by_number(session, old_version_number)
        new_v = get_version_by_number(session, new_version_number)

        if not old_v or not new_v:
            raise ValueError("Versija nerasta DB.")

        old_terms = {t.term_code: t for t in get_terms_by_version(session, old_v.id)}
        new_terms = {t.term_code: t for t in get_terms_by_version(session, new_v.id)}

        changes = []

        for code, new_term in new_terms.items():
            if code not in old_terms:
                changes.append(MedDRAChange(
                    old_version_id=old_v.id,
                    new_version_id=new_v.id,
                    term_code=code,
                    change_type=TermChangeType.ADDED,
                    new_text=new_term.term_text,
                ))
            elif old_terms[code].term_text != new_term.term_text:
                score = bert.similarity(old_terms[code].term_text, new_term.term_text)
                changes.append(MedDRAChange(
                    old_version_id=old_v.id,
                    new_version_id=new_v.id,
                    term_code=code,
                    change_type=TermChangeType.MODIFIED,
                    old_text=old_terms[code].term_text,
                    new_text=new_term.term_text,
                    similarity_score=score,
                ))

        for code, old_term in old_terms.items():
            if code not in new_terms:
                changes.append(MedDRAChange(
                    old_version_id=old_v.id,
                    new_version_id=new_v.id,
                    term_code=code,
                    change_type=TermChangeType.DELETED,
                    old_text=old_term.term_text,
                ))

            session.add_all(changes)
            session.commit()
            return changes
    except Exception as e:
        session.rollback()
        raise RuntimeError(f"Klaida aptinkant pakeitimus: {e}")

    
            
