from sentence_transformers import SentenceTransformer
from src.models.base_model import BaseMLModel
from config import config

class BertModel(BaseMLModel):

    def load_model(self) -> None:
        try:
            self.model = SentenceTransformer(
                config.BERT_MODEL_NAME,
                devices=str(self.device)
            )
        except Exception as e:
            raise RuntimeError(f"Klaida ikeliant BERT modeli: {e}")

    def predict(self, texts: list[str]) ->list:
        if not self.is_loaded():
            self.load_model()
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings.tolist()
        except Exception as e:
            raise RuntimeError(f"Klaida skaiciuojant embeddings: {e}")

    def similarity(self, text1: str, text2: str) -> float:
        import torch
        if not self.is_loaded():
            self.load_model()
        emb = self.model.encode([text1, text2], convert_to_tensor=True)
        score = torch.nn.functional.cosine_similarity(
            emb[0].unsqueeze(0), emb[1].unsqueeze(0)
        )
        return float(score)