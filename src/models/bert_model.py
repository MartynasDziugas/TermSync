import hashlib
import threading
from collections import OrderedDict

from config import config
from src.models.base_model import BaseMLModel


class BertModel(BaseMLModel):
    """SentenceTransformer kraunamas tik load_model() – greitesnis Flask paleidimas."""

    def __init__(self) -> None:
        super().__init__()
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_max: int = getattr(
            config, "EMBEDDING_CACHE_MAX_ITEMS", 20_000
        )

    def load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                config.BERT_MODEL_NAME,
                device=str(self.device),
            )
        except Exception as e:
            raise RuntimeError(f"Klaida ikeliant BERT modeli: {e}")

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_put(self, key: str, vec: list[float]) -> None:
        self._cache[key] = vec
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def predict(self, texts: list[str]) -> list[list[float]]:
        if not self.is_loaded():
            self.load_model()
        try:
            with self._cache_lock:
                keys = [self._cache_key(t) for t in texts]
                missing_texts: list[str] = []
                missing_keys: list[str] = []
                seen_m: set[str] = set()
                for t, k in zip(texts, keys, strict=True):
                    if k in self._cache:
                        self._cache.move_to_end(k)
                    elif k not in seen_m:
                        seen_m.add(k)
                        missing_texts.append(t)
                        missing_keys.append(k)

            if missing_texts:
                new_emb = self.model.encode(
                    missing_texts,
                    convert_to_tensor=True,
                )
                rows = new_emb.tolist()
                with self._cache_lock:
                    for mk, vec in zip(missing_keys, rows, strict=True):
                        self._cache_put(mk, vec)

            out: list[list[float]] = []
            with self._cache_lock:
                for k in keys:
                    out.append(list(self._cache[k]))
            return out
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
