from pathlib import Path
from sqlalchemy.orm import Session
from src.database.models import DocumentTemplate, TemplateSegment
from src.parsers.docx_parser import DocxParser


class TemplateImporter:
    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.parser = DocxParser(file_path)

    def validate(self) -> bool:
        return self.parser.validate()

    def import_template(self, session: Session, name: str, language: str = "LT") -> DocumentTemplate:
        try:
            if not self.validate():
                raise ValueError(f"Netinkamas failas: {self.file_path}")
            

            template = DocumentTemplate(
                name=name,
                file_path=str(self.file_path),
                language=language,
            )
            session.add(template)
            session.flush()

            segments = self.parser.extract_segments()
            for i, text in enumerate(segments):
                segment = TemplateSegment(
                    template_id=template.id,
                    segment_index=i,
                    source_text=text,
                )
                session.add(segment)

            session.commit()
            return template
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Klaida importuojant sablona: {e}")
            
