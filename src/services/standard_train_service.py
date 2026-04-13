"""
Standarto porų .docx: teisingos + dirbtinės blogos poros, BERT požymiai,
80/20 train/test, SVM + greitas MLP, tada SVM per visa imtį ir įrašas į diską.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from config import config
from src.models.bert_model import BertModel
from src.models.mlp_classifier import MLPClassifier
from src.models.svm_bundle import (
    LABEL_MATCH,
    LABEL_NO_MATCH,
    StandardPairBundle,
    new_svm_pair_classifier,
    pack_bundle,
    unpack_bundle_dict,
)
from src.parsers.docx_parser import DocxParser

_bert_singleton: BertModel | None = None


def _get_bert() -> BertModel:
    global _bert_singleton
    if _bert_singleton is None:
        _bert_singleton = BertModel()
    return _bert_singleton


def extract_aligned_paragraph_pairs(
    path_left: Path,
    path_right: Path,
) -> tuple[list[tuple[str, str]], int, int]:
    pl = DocxParser(path_left)
    pr = DocxParser(path_right)
    if not pl.validate() or not pr.validate():
        raise ValueError("Abu failai turi būti galiojantys .docx.")
    left = pl.extract_segments()
    right = pr.extract_segments()
    if not left or not right:
        raise ValueError("Bent viename faile nerasta ne tuščių pastraipų.")
    n = min(len(left), len(right))
    pairs = [(left[i], right[i]) for i in range(n)]
    return pairs, len(left), len(right)


def _build_xy_from_pairs(
    aligned: list[tuple[str, str]],
    rng: np.random.Generator,
) -> tuple[list[str], list[str], list[str]]:
    n = len(aligned)
    left_m, right_m = zip(*aligned, strict=True)
    left_m = list(left_m)
    right_m = list(right_m)
    labels_m = [LABEL_MATCH] * n

    left_b: list[str] = []
    right_b: list[str] = []
    for i in range(n):
        choices = [j for j in range(n) if j != i]
        if not choices:
            break
        j = int(rng.choice(choices))
        left_b.append(left_m[i])
        right_b.append(right_m[j])
    labels_b = [LABEL_NO_MATCH] * len(left_b)

    texts_left = left_m + left_b
    texts_right = right_m + right_b
    labels = labels_m + labels_b
    return texts_left, texts_right, labels


def _pair_feature_matrix(
    texts_left: list[str],
    texts_right: list[str],
    bert: BertModel,
) -> np.ndarray:
    emb_l = np.array(bert.predict(texts_left), dtype=np.float64)
    emb_r = np.array(bert.predict(texts_right), dtype=np.float64)
    return np.hstack([emb_l, emb_r])


def _runs_dir(job_id: str) -> Path:
    d = config.STANDARD_RUNS_FOLDER / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_standard_train_job(
    path_left: Path,
    path_right: Path,
    job_id: str,
    *,
    display_src_name: str | None = None,
    display_tgt_name: str | None = None,
) -> None:
    from src.services.pair_train_job_store import update_job

    def _upd(progress: int, message: str) -> None:
        update_job(job_id, progress=progress, message=message, status="running")

    try:
        config.STANDARD_RUNS_FOLDER.mkdir(parents=True, exist_ok=True)
        _upd(5, "Skaitomi standarto .docx ir poruojamos pastraipos…")
        aligned, left_seg_n, right_seg_n = extract_aligned_paragraph_pairs(
            path_left, path_right
        )
        if len(aligned) < 6:
            raise ValueError(
                "Reikia bent 6 sutampančių ne tuščių pastraipų poroje (min(ilgis_kairė, ilgis_dešinė))."
            )

        rng = np.random.default_rng(42)
        texts_left, texts_right, labels = _build_xy_from_pairs(aligned, rng)

        _upd(22, f"Standarto porų: {len(aligned)}; mokymo eilučių: {len(labels)} (su dirbtinėmis blogomis poromis).")
        bert = _get_bert()
        _upd(38, "BERT embeddingai (kešuojami)…")
        X = _pair_feature_matrix(texts_left, texts_right, bert)

        _upd(52, "80/20 train / test…")
        y = np.array(labels)
        strat = None
        _, counts = np.unique(y, return_counts=True)
        if len(labels) >= 20 and int(counts.min()) >= 2:
            strat = y

        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=strat,
        )
        y_tr_list = y_tr.tolist()
        y_te_list = y_te.tolist()

        _upd(65, "SVM (test rinkinys)…")
        svm = new_svm_pair_classifier()
        svm.fit(X_tr, y_tr)
        svm_acc = float(svm.score(X_te, y_te))

        _upd(78, f"MLP ({config.QUICK_MLP_EPOCHS} epochų, test)…")
        mlp = MLPClassifier(
            input_dim=X.shape[1],
            hidden_dim=128,
            dropout=0.2,
        )
        mlp.fit(
            X_tr,
            y_tr_list,
            epochs=config.QUICK_MLP_EPOCHS,
            lr=1e-3,
        )
        mlp_acc = float(mlp.score(X_te, y_te))

        _upd(88, "SVM per visą imtį (įrašymas)…")
        svm_full = new_svm_pair_classifier()
        svm_full.fit(X, y)

        run_dir = _runs_dir(job_id)
        bundle_path = run_dir / "bundle.joblib"
        bundle = pack_bundle(svm_full, aligned, int(X.shape[1]))
        joblib.dump(bundle, bundle_path)
        artifact_abs = str(bundle_path.resolve())

        result = {
            "n_aligned_pairs": len(aligned),
            "n_train_rows": int(X_tr.shape[0]),
            "n_test_rows": int(X_te.shape[0]),
            "embedding_dim_per_side": int(X.shape[1] // 2),
            "svm_test_accuracy": round(svm_acc, 4),
            "mlp_test_accuracy": round(mlp_acc, 4),
            "left_paragraph_count": left_seg_n,
            "right_paragraph_count": right_seg_n,
            "truncated_to_min_length": len(aligned) < max(left_seg_n, right_seg_n),
            "quick_mlp_epochs": config.QUICK_MLP_EPOCHS,
            "artifact_path": artifact_abs,
            "run_id": job_id,
        }

        from src.services.persist_service import persist_training_run

        persist_training_run(
            job_id=job_id,
            standard_src_name=display_src_name or path_left.name,
            standard_tgt_name=display_tgt_name or path_right.name,
            aligned=aligned,
            result=result,
            artifact_path=artifact_abs,
        )

        _upd(100, "Baigta. Modelis ir DB įrašyti — galite eiti į Tikrinimą.")
        update_job(
            job_id,
            status="done",
            progress=100,
            message="Baigta.",
            result=result,
            error=None,
        )
    except Exception as e:
        update_job(
            job_id,
            status="error",
            progress=0,
            message="Klaida.",
            result=None,
            error=str(e),
        )


def load_latest_bundle() -> tuple[StandardPairBundle, str] | None:
    """Grąžina (SVM bundle + standarto poros, job_id) arba None."""
    try:
        from src.services.persist_service import get_active_training_run

        tr = get_active_training_run()
        if tr is not None:
            bp = Path(tr.artifact_path)
            if bp.is_file():
                raw = joblib.load(bp)
                return unpack_bundle_dict(raw), tr.job_id
    except Exception:
        pass
    p = config.LATEST_STANDARD_RUN_FILE
    if not p.is_file():
        return None
    job_id = p.read_text(encoding="utf-8").strip()
    if not job_id:
        return None
    bundle_path = config.STANDARD_RUNS_FOLDER / job_id / "bundle.joblib"
    if not bundle_path.is_file():
        return None
    raw = joblib.load(bundle_path)
    return unpack_bundle_dict(raw), job_id
