from pathlib import Path 
from docx import Document

class DocxParser:

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def validate(self) -> bool:
        try:
            return self.file_path.exists() and self.file_path.suffix == ".docx"
        except Exception:
            return False

    def extract_segments(self) -> list[str]:
        try:
            doc = Document(self.file_path)
            segments = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    segments.append(text)
            return segments
        except Exception as e:
            raise RuntimeError(f"Klaida skaitant .docx faila: {e}")
        