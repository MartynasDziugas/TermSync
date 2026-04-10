from pathlib import Path
from src.parsers.base import TerminologyImporter

class MedDRAImporter(TerminologyImporter):

    def validate(self) -> bool:
        try:
            return self.file_path.exists() and self.file_path.suffix == ".asc"
        except Exception:
            return False

    def parse(self) -> list[dict]:
        terms = []
        try:
            with open(self.file_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.strip().split("$")
                    if len(parts) >=2:
                        terms.append({
                            "term_code": parts[0].strip(),
                            "term_text": parts[1].strip(),
                        })
        except Exception as e:
            raise RuntimeError(f"Klaida skaitant MedDRA faila: {e}")
        return terms
        
    def get_versions(self) -> str:
         # MedDRA versija nustatoma iš aplanko pavadinimo arba failo pavadinimo
        # pvz. "MedDRA_27_1_Lithuanian" -> "27.1"
        name = self.file_path.parent.name
        parts = name.split("_")
        if len(parts) >= 3:
            return f"{parts[1]}.{parts[2]}"
        return "unknown"
