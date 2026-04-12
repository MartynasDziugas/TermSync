from src.database.connection import get_session, init_db
from src.models.bert_model import BertModel
from src.services.experiment_service import save_experiment
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import numpy as np

init_db()
bert = BertModel()

SAMPLE_TEXTS = [
    "adverse reaction", "clinical trial", "headache", "nausea",
    "pharmacovigilance", "contraindication", "dosage", "side effect",
    "drug interaction", "hypersensitivity"
]

LABELS = ["ae", "ct", "ae", "ae", "pv", "ci", "dos", "ae", "di", "ae"]

embeddings = bert.predict(SAMPLE_TEXTS)
X = np.array(embeddings)
le = LabelEncoder()
y = le.fit_transform(LABELS)

experiments = [
    {"kernel": "rbf",     "C": 1.0,  "gamma": "scale"},
    {"kernel": "rbf",     "C": 0.5,  "gamma": "scale"},
    {"kernel": "rbf",     "C": 2.0,  "gamma": "scale"},
    {"kernel": "rbf",     "C": 5.0,  "gamma": "scale"},
    {"kernel": "rbf",     "C": 10.0, "gamma": "scale"},
    {"kernel": "rbf",     "C": 1.0,  "gamma": "auto"},
    {"kernel": "rbf",     "C": 2.0,  "gamma": "auto"},
    {"kernel": "rbf",     "C": 5.0,  "gamma": "auto"},
    {"kernel": "linear",  "C": 1.0,  "gamma": "scale"},
    {"kernel": "linear",  "C": 0.5,  "gamma": "scale"},
    {"kernel": "linear",  "C": 2.0,  "gamma": "scale"},
    {"kernel": "linear",  "C": 5.0,  "gamma": "scale"},
    {"kernel": "poly",    "C": 1.0,  "gamma": "scale"},
    {"kernel": "poly",    "C": 2.0,  "gamma": "scale"},
    {"kernel": "poly",    "C": 5.0,  "gamma": "scale"},
    {"kernel": "sigmoid", "C": 1.0,  "gamma": "scale"},
    {"kernel": "sigmoid", "C": 0.5,  "gamma": "scale"},
    {"kernel": "sigmoid", "C": 2.0,  "gamma": "scale"},
    {"kernel": "rbf",     "C": 1.0,  "gamma": 0.1},
    {"kernel": "rbf",     "C": 1.0,  "gamma": 0.01},
    {"kernel": "rbf",     "C": 2.0,  "gamma": 0.1},
    {"kernel": "rbf",     "C": 2.0,  "gamma": 0.01},
    {"kernel": "rbf",     "C": 5.0,  "gamma": 0.1},
    {"kernel": "rbf",     "C": 5.0,  "gamma": 0.01},
    {"kernel": "linear",  "C": 10.0, "gamma": "scale"},
]

with get_session() as session:
    for i, params in enumerate(experiments):
        svm = SVC(
            kernel=params["kernel"],
            C=params["C"],
            gamma=params["gamma"],
            probability=True
        )
        scores = cross_val_score(svm, X, y, cv=2, scoring="accuracy")
        metrics = {
            "accuracy_mean": round(float(scores.mean()), 4),
            "accuracy_std": round(float(scores.std()), 4),
        }
        save_experiment(
            session=session,
            experiment_name=f"exp_{i+1:02d}",
            model_type="SVM",
            hyperparameters=params,
            metrics=metrics,
            dataset_description="10 medical terms, BERT embeddings",
        )
        print(f"exp_{i+1:02d} | {params} | {metrics}")