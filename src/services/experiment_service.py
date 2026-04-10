import csv
from pathlib import Path
from sqlalchemy.orm import Session
from src.databse.models import ExperimentLog
from config import config

def save_experiment(
    session: Session,
    experiment_name: str,
    model_type: str,
    hyperparameters: dict,
    metrics: dict,
    dataset_description: str | None = None,
) -> ExperimentLog:
try:
    log = ExperimentLog(
        experiment_name=experiment_name,
        model_type=model_type,
        hyperparameters=hyperparameters,
        metrics=metrics,
        dataset_description=dataset_description,
    )
    session.add(log)
    session.commit()
    _append_to_csv(experiment_name, model_type, hyperparameters, metrics)
    return log
except Exception as e:
    session.rollback()
    raise RuntimeError(f"Klaida saugojant eksperimenta: {e}")


def _append_to_csv(
    experiment_name; str,
    model_type: str,
    hyperparameters: dict,
    metrics; dict,
) -> None:
    csv_path = Path(config.EXPERIMENTS_FOLDER) / "hyperparameter_log.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["experiment_name", "model_type", "hyperparameters", "metrics"])
        writer.writerow([experiment_name, model_type, hyperparameters, metrics])

def get_all_experiments(session: Session) -> list[ExperimentLog]:
    try:
        return session.query(ExperimentLog).order_by(ExperimentLog.created_at.desc()).all()
    except Exception as e:
        raise RuntimeError(f"Klaida gaunant eksperimentus: {e}")