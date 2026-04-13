"""
Pilnas neuroninis klasifikatorius (PyTorch MLP) ant fiksuoto ilgio vektorių.

Tas pats uždavinys kaip ir SVMModel: iš embeddingų prognozuoti etiketę.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn


class MLPClassifier:
    """Daugiaklasė klasifikacija: įėjimas = embeddingų vektorius, išėjimas = klasė."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.label_encoder = LabelEncoder()
        self._net: nn.Module | None = None
        self._num_classes: int = 0

    def _build(self, num_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, num_classes),
        ).to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y_labels: list[str],
        *,
        epochs: int = 120,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        y = self.label_encoder.fit_transform(np.array(y_labels))
        self._num_classes = int(len(self.label_encoder.classes_))
        self._net = self._build(self._num_classes)
        assert self._net is not None

        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.long, device=self.device)
        opt = torch.optim.AdamW(self._net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        self._net.train()
        n = Xt.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            logits = self._net(Xt[perm])
            loss = loss_fn(logits, yt[perm])
            opt.zero_grad()
            loss.backward()
            opt.step()

        self._net.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("MLPClassifier: pirma iskvieskite fit().")
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.tensor(X, dtype=torch.float32, device=self.device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()
        return self.label_encoder.inverse_transform(pred)

    def score(self, X: np.ndarray, y_labels: list[str]) -> float:
        pred = self.predict(X)
        y = np.array(y_labels)
        return float((pred == y).mean())
