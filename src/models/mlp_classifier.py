"""
Pilnas neuroninis klasifikatorius (PyTorch MLP) ant fiksuoto ilgio vektorių.

Tas pats uždavinys kaip ir SVMModel: iš embeddingų prognozuoti etiketę.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch import nn


def _activation_module(name: str) -> nn.Module:
    n = (name or "relu").lower()
    if n == "tanh":
        return nn.Tanh()
    if n == "gelu":
        return nn.GELU()
    if n == "silu":
        return nn.SiLU()
    return nn.ReLU()


class MLPClassifier:
    """Daugiaklasė klasifikacija: įėjimas = embeddingų vektorius, išėjimas = klasė."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        *,
        hidden_sizes: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.2,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.input_dim = input_dim
        self.dropout = dropout
        self.activation_name = activation
        if hidden_sizes is None:
            self.hidden_sizes = [hidden_dim, hidden_dim]
        else:
            self.hidden_sizes = [max(4, int(x)) for x in hidden_sizes]
        self.label_encoder = LabelEncoder()
        self._net: nn.Module | None = None
        self._num_classes: int = 0
        self.final_train_loss: float | None = None
        self.loss_history: list[float] = []
        self.val_accuracy_history: list[float] = []
        self.val_f1_history: list[float] = []

    def _build(self, num_classes: int) -> nn.Module:
        layers: list[nn.Module] = []
        prev = self.input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(_activation_module(self.activation_name))
            layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        return nn.Sequential(*layers).to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y_labels: list[str],
        *,
        epochs: int = 120,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        X_val: np.ndarray | None = None,
        y_val: list[str] | None = None,
    ) -> None:
        self.loss_history = []
        self.val_accuracy_history = []
        self.val_f1_history = []

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
            self.loss_history.append(float(loss.detach().cpu().item()))

            if X_val is not None and y_val is not None and len(y_val) == len(X_val):
                self._net.eval()
                with torch.no_grad():
                    acc = float(self.score(X_val, y_val))
                    pred_v = self.predict(X_val)
                    f1v = float(
                        f1_score(
                            np.array(y_val),
                            pred_v,
                            average="weighted",
                            zero_division=0,
                        )
                    )
                self.val_accuracy_history.append(acc)
                self.val_f1_history.append(f1v)
                self._net.train()

        self._net.eval()
        with torch.no_grad():
            self.final_train_loss = float(loss_fn(self._net(Xt), yt).item())

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
