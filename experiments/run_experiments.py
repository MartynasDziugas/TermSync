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
        

# --- BERT vs SVM palyginimas ---
from sklearn.metrics.pairwise import cosine_similarity

bert_scores = []
for i in range(len(SAMPLE_TEXTS)):
    for j in range(i + 1, len(SAMPLE_TEXTS)):
        score = float(cosine_similarity(
            X[i].reshape(1, -1), X[j].reshape(1, -1)
        )[0][0])
        bert_scores.append(score)

bert_mean = round(sum(bert_scores) / len(bert_scores), 4)
bert_std = round(float(np.std(bert_scores)), 4)

with get_session() as session:
    save_experiment(
        session=session,
        experiment_name="bert_baseline",
        model_type="BERT",
        hyperparameters={"model": "paraphrase-multilingual-MiniLM-L12-v2"},
        metrics={
            "cosine_similarity_mean": bert_mean,
            "cosine_similarity_std": bert_std,
        },
        dataset_description="10 medical terms, pairwise cosine similarity",
    )
    print(f"BERT baseline | similarity_mean={bert_mean} | std={bert_std}")