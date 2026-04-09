from abc import ABC, abstractmethod
import torch

class BaseMLModel(ABC):

    def __init__(self) -> None:
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """Ikelia modeli i atminti."""
        pass

    @abstractmethod
    def predict(self, texts: list[str]) -> list:
        """Grazina modelio rezultatus pagal pateiktus tekstus."""
        pass
    
    def is_loaded(self) -> bool:
        return self.model is not None

    