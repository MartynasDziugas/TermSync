from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import numpy as np
from src.models.base_model import BaseMLModel

class SVMModel(BaseMLModel):

    def __init__(self) -> None:
        super().__init__()
        self.label_encoder = LabelEncoder()

    def load_model(self) -> None:
        try:
            self.model = SVC(kernel="rbf", probability=True)
        except Exception as e:
            raise RuntimeError(f"Klaida kuriant SVM modeli: {e}")

    def train(self, embeddings: list[list[float]], labels: list[str]) -> None:
        if not self.is_loaded():
            self.load_model()
        try:
            X = np.array(embeddings)
            y = self.label_encoder.fit_transform(labels)
            self.model.fit(X, y)
        except Exception as e:
            raise RuntimeError(f"Klaida treniruojant SVM modeli: {e}")

    def predict(self, texts: list[list[float]]) -> list:
        if not self.is_loaded():
            raise RuntimeError("SVM modelis neikeltas. Pirma iskvieskite train().")
        try:
            X = np.array(texts)
            encoded = self.model.predict(X)
            return self.label_encoder.inverse_transform(encoded).tolist()
        except Exception as e:
            raise RuntimeError(f"Klaida klasifikuojant: {e}")
            