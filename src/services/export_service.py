from pathlib import Path
from sqlalchemy.orm import Session
from docx import Document
from src.database.queries import get_changes_between_versions
from src.database.models import TermChangeType

def export_to_tmx(
    session: Session,
    old_version_id: int,
    new_version_id: int,
    output_path: str | Path,
) -> Path:
try:
    changes = get_changes_between_versions(session, old_version_id, new_version_id)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<tmx version="1.4"><body>']
    for change in changes:
        if change.new_text:
            lines.append(f'  <tu>')
                lines.append(f'    <tuv xml:lang="lt"><seg>{change.new_text}</seg></tuv>')
                lines.append(f'  </tu>')
        lines.append('</body></tmx>')'

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Klaida eksportuojant i TMX; {e}")


def export_to_docx(
    session: Session,
    old_version_id: int,
    new_version_id: int,
    output_path: str | Path,
) -> Path:
    try:

    changes = get_changes_between_versions(session, old_version_id, new_version_id)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()
    doc.add_heading("MedDRA pakeitimų ataskaita", level=1)
        doc.add_heading("Pridėti terminai", level=2)
        for c in changes:
            if c.change_type == TermChangeType.ADDED:
                doc.add_paragraph(f"+ {c.new_text}")
        doc.add_heading("Ištrinti terminai", level=2)
        for c in changes:
            if c.change_type == TermChangeType.DELETED:
                doc.add_paragraph(f"- {c.old_text}")
        doc.add_heading("Pakeisti terminai", level=2)
        for c in changes:
            if c.change_type == TermChangeType.MODIFIED:
                doc.add_paragraph(
                    f"{c.old_text} → {c.new_text} (panašumas: {c.similarity_score:.2f})"
                )
        doc.save(output_path)
        return output_path
    except Exception as e:
        raise RuntimeError(f"Klaida eksportuojant į DOCX: {e}")
