from abc import ABC, abstractmethod
from pathlib import Path

class TerminologyImporter(ABC):

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    @abstractmethod
    def validate(self) -> bool:
        """Patikrina ar failas tinkamas importuoti."""
        pass

    @abstractmethod
    def parse(self) -> list[dict]:
        """Nuskaito faila ir grazina terminu sarasa."""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Grazina versijos numeri is failo."""
        pass
        
