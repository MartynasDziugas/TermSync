import xml.ertree.ElementTree as ET
from pathlib import Path
from src.parsers.base import TerminologyImporter

class TMXImporter(TerminologyImporter):
    
    def validate(self) -> bool:
        try:
            return self.file_path.exists() and self.file_path.suffix == ".tmx"
        except Exception:
            return False

    def parse(self) -> list[dict]:
        terms = []
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()
            for tu in root.iter("tu"):
                entry = {}
                for tuv in tu.findall("tuv"):
                    lang = tuv.get("{http://www.w3.org/XML/1998/namespace}lang", "")
                    seg = tuv.find("seg")
                    if seg is not None and seg.text:
                        entry[lang.lower()] = seg.text.strip()
                if entry:
                    terms.append(entry)
            except Ex eption as e:
                raise RuntimeError(f"Klaida skaitant TMX faila: {e}")
            return terms
        
    def get_version(self) -> str:
        return self.file_path.stem



