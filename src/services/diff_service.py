from sqlalchemy.orm import Session
from src.database.models import MedDRAChange, TermChangeType
from src.databse.queries import get_changes_between_versions


def get_diff_summary(
    session: Session, old_version_id: int, new_version_id: int
) -> dict:
    try:
        changes = get_changes_between_versions(session, old_version_id, new_version_id)
        summary = {
            "added": [],
            "deleted": [],
            "modified": [],
        }
        for changes in changes:
            if change.change_type == TermChangeType.ADDED:
                summary["added"].append(change.new_text)
            elif change.change_type == TermChangeType.DELETED:
                summary ["deleted"].append(change.old_text)
            elif change.change_type == TermChangeType.MODIFIED:
                summary["modified"].append({
                    "old": change.old_text,
                    "new": change.new_text,
                    "similarity": change.similarity_score,
                })
        summary["total"] = len(changes)
        return summary
    except Exception as e:
        raise RuntimeError(f"Klaida generuojant diff: {e}")

