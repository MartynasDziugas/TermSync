from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


def _paragraph_xml_text(paragraph) -> str:
    """
    Surenka visą pastraipos tekstą iš w:t elementų (įskaitant hipernuorodų ir laukų vidų),
    kur para.text kartais būna trumpesnis.
    """
    parts: list[str] = []
    for node in paragraph._p.iter(qn("w:t")):
        if node.text:
            parts.append(node.text)
    return "".join(parts)


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
            segments: list[str] = []
            for para in doc.paragraphs:
                text = _paragraph_xml_text(para).strip()
                if not text:
                    text = (para.text or "").strip()
                if text:
                    segments.append(text)
            return segments
        except Exception as e:
            raise RuntimeError(f"Klaida skaitant .docx faila: {e}")
