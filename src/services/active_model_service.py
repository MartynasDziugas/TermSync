"""Aktyvaus porų klasifikatoriaus (SVM / MLP) pasirinkimas vertėjo peržiūrai."""

from __future__ import annotations

from pathlib import Path

import joblib
from sqlalchemy import select, update

from src.database.connection import get_session
from src.database.models import TrainingRun
from src.services.persist_service import get_active_training_run


def set_active_model(model_type: str, *, job_id: str) -> str:
    """
    Įrašo `active_model_type` aktyviam mokymo įrašui, jei `job_id` sutampa su aktyviu.
    MLP atveju tikrina, ar bundle turi MLP objektą.
    """
    mt = (model_type or "").strip().lower()
    if mt not in ("svm", "mlp"):
        raise ValueError("model_type turi būti 'svm' arba 'mlp'.")

    active = get_active_training_run()
    if active is None:
        raise ValueError("Nėra aktyvaus standarto mokymo įrašo DB.")
    if active.job_id != job_id:
        raise ValueError(
            "Galima keisti tik dabartinį aktyvų mokymą: job_id nesutampa su aktyviu įrašu."
        )

    artifact = Path(active.artifact_path)
    if not artifact.is_file():
        raise ValueError(f"Artefaktas nerastas: {artifact}")

    if mt == "mlp":
        raw = joblib.load(artifact)
        if not isinstance(raw, dict) or raw.get("mlp") is None:
            raise ValueError(
                "Šiame bundle nėra MLP (per senas artefaktas). Paleiskite standarto mokymą iš naujo."
            )

    with get_session() as session:
        row = session.scalars(
            select(TrainingRun).where(TrainingRun.job_id == job_id)
        ).first()
        if row is None:
            raise ValueError(f"Nerastas TrainingRun job_id={job_id!r}.")
        session.execute(
            update(TrainingRun)
            .where(TrainingRun.id == row.id)
            .values(active_model_type=mt)
        )
    return mt
