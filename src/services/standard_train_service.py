"""
Standarto porų .docx: teisingos + dirbtinės blogos poros, BERT požymiai,
80/20 train/test, SVM + greitas MLP, tada SVM per visa imtį ir įrašas į diską.

Encoderis — užšaldytas SentenceTransformer; iš JSON hiperparametrų valdomi tik SVM ir MLP.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score, hinge_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import BASE_DIR, config
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


def _parse_lr_token(s: object) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    text = str(s).strip().lower().replace(" ", "")
    return float(text)


def _parse_hidden_layer_sizes(spec: str) -> list[int]:
    parts = [p.strip() for p in str(spec).replace(";", ",").split(",") if p.strip()]
    out = [max(8, int(p)) for p in parts]
    return out if out else [128, 128]


def _resolve_hyperparams(hyperparams: dict | None) -> dict[str, object]:
    """Numatytieji = ankstesnis fiksuotas elgesys; iš JSON — tik SVM ir MLP (encoder nekeičiamas)."""
    out: dict[str, object] = {
        "mlp_lr": 1e-3,
        "mlp_epochs": int(config.QUICK_MLP_EPOCHS),
        "svm_c": 1.0,
        "svm_kernel": "rbf",
        "mlp_hidden_sizes": [128, 128],
        "mlp_activation": "relu",
        "encoder_note": (
            "SentenceTransformer embeddingai užšaldyti; hiperparametrai valdo tik SVM ir MLP."
        ),
    }
    if not hyperparams:
        return out
    svm = hyperparams.get("svm") or {}
    mlp = hyperparams.get("mlp") or {}
    if mlp.get("learningRate") is not None:
        try:
            out["mlp_lr"] = max(float(_parse_lr_token(mlp["learningRate"])), 1e-7)
        except (TypeError, ValueError):
            pass
    if mlp.get("epochs") is not None:
        try:
            out["mlp_epochs"] = max(1, min(500, int(mlp["epochs"])))
        except (TypeError, ValueError):
            pass
    try:
        out["svm_c"] = float(svm.get("C", 1.0))
    except (TypeError, ValueError):
        out["svm_c"] = 1.0
    sk = str(svm.get("kernel", "rbf")).lower()
    out["svm_kernel"] = sk if sk in ("rbf", "linear", "poly", "sigmoid") else "rbf"
    if mlp.get("hiddenLayerSizes"):
        try:
            out["mlp_hidden_sizes"] = _parse_hidden_layer_sizes(str(mlp["hiddenLayerSizes"]))
        except ValueError:
            pass
    out["mlp_activation"] = str(mlp.get("activation", "relu")).lower()
    return out


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


def _append_hyperparam_log(job_id: str, hyperparams: dict | None, result: dict) -> None:
    if hyperparams is None:
        return
    log_path = BASE_DIR / "experiments" / "hyperparameter_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "hyperparams": hyperparams,
        "metrics": {k: result[k] for k in result if k in ("svm_test_accuracy", "mlp_test_accuracy", "mlp_train_loss")},
    }
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(line, ensure_ascii=False) + "\n")


def run_standard_train_job(
    path_left: Path,
    path_right: Path,
    job_id: str,
    *,
    display_src_name: str | None = None,
    display_tgt_name: str | None = None,
    hyperparams: dict | None = None,
) -> dict | None:
    from src.services.pair_train_job_store import ensure_job_slot, update_job

    ensure_job_slot(job_id)

    def _upd(progress: int, message: str) -> None:
        update_job(job_id, progress=progress, message=message, status="running")

    hp = _resolve_hyperparams(hyperparams)
    mlp_epochs = int(hp["mlp_epochs"])
    mlp_lr = float(hp["mlp_lr"])
    svm_c = float(hp["svm_c"])
    svm_kernel = str(hp["svm_kernel"])
    mlp_hidden: list[int] = list(hp["mlp_hidden_sizes"])  # type: ignore[arg-type]
    mlp_act = str(hp["mlp_activation"])

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
        _upd(38, "BERT embeddingai (SentenceTransformer, kešuojami)…")
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

        _upd(65, f"SVM (kernel={svm_kernel}, C={svm_c})…")
        svm = new_svm_pair_classifier(C=svm_c, kernel=svm_kernel)
        svm.fit(X_tr, y_tr)
        svm_acc = float(svm.score(X_te, y_te))
        svm_pred_te = svm.predict(X_te)
        svm_f1 = float(
            f1_score(y_te, svm_pred_te, average="weighted", zero_division=0)
        )
        y_tr_np = np.asarray(y_tr)
        _svm_le = LabelEncoder().fit(svm.classes_)
        y_tr_enc = _svm_le.transform(y_tr_np)
        svm_hinge_train = float(
            hinge_loss(
                y_tr_enc,
                svm.decision_function(X_tr),
                labels=np.arange(len(svm.classes_)),
            )
        )

        _upd(78, f"MLP ({mlp_epochs} epochų, hidden={mlp_hidden}, act={mlp_act})…")
        mlp = MLPClassifier(
            input_dim=X.shape[1],
            hidden_sizes=mlp_hidden,
            activation=mlp_act,
            dropout=0.2,
        )
        mlp.fit(
            X_tr,
            y_tr_list,
            epochs=mlp_epochs,
            lr=mlp_lr,
            X_val=X_te,
            y_val=y_te_list,
        )
        mlp_acc = float(mlp.score(X_te, y_te_list))
        mlp_pred_te = mlp.predict(X_te)
        mlp_f1 = float(
            f1_score(np.asarray(y_te_list), mlp_pred_te, average="weighted", zero_division=0)
        )

        _upd(88, "SVM per visą imtį (įrašymas)…")
        svm_full = new_svm_pair_classifier(C=svm_c, kernel=svm_kernel)
        svm_full.fit(X, y)

        _upd(92, "MLP per visą imtį (įrašymas į bundle)…")
        mlp_full = MLPClassifier(
            input_dim=X.shape[1],
            hidden_sizes=mlp_hidden,
            activation=mlp_act,
            dropout=0.2,
        )
        mlp_full.fit(X, y.tolist(), epochs=mlp_epochs, lr=mlp_lr)

        run_dir = _runs_dir(job_id)
        bundle_path = run_dir / "bundle.joblib"
        bundle = pack_bundle(svm_full, aligned, int(X.shape[1]), mlp=mlp_full)
        joblib.dump(bundle, bundle_path)
        artifact_abs = str(bundle_path.resolve())

        n_ep = len(mlp.loss_history)
        ep_axis = list(range(1, n_ep + 1)) if n_ep else []
        result = {
            "n_aligned_pairs": len(aligned),
            "n_train_rows": int(X_tr.shape[0]),
            "n_test_rows": int(X_te.shape[0]),
            "embedding_dim_per_side": int(X.shape[1] // 2),
            "svm_test_accuracy": round(svm_acc, 4),
            "svm_test_f1": round(svm_f1, 4),
            "mlp_test_accuracy": round(mlp_acc, 4),
            "mlp_test_f1": round(mlp_f1, 4),
            "mlp_train_loss": round(float(mlp.final_train_loss or 0.0), 6),
            "svm_train_hinge": round(svm_hinge_train, 4),
            "svm_charts": {
                "labels": ["Test"],
                "accuracy": [round(svm_acc, 4)],
                "f1": [round(svm_f1, 4)],
                "loss": [round(svm_hinge_train, 4)],
            },
            "mlp_charts": {
                "epochs": ep_axis,
                "accuracy": [round(float(x), 4) for x in mlp.val_accuracy_history],
                "f1": [round(float(x), 4) for x in mlp.val_f1_history],
                "loss": [round(float(x), 6) for x in mlp.loss_history],
            },
            "active_model_type": "svm",
            "left_paragraph_count": left_seg_n,
            "right_paragraph_count": right_seg_n,
            "truncated_to_min_length": len(aligned) < max(left_seg_n, right_seg_n),
            "quick_mlp_epochs": mlp_epochs,
            "artifact_path": artifact_abs,
            "run_id": job_id,
            "svm_C": svm_c,
            "svm_kernel": svm_kernel,
            "mlp_hidden_sizes": mlp_hidden,
            "mlp_activation": mlp_act,
            "mlp_learning_rate": mlp_lr,
            "mlp_epochs_used": mlp_epochs,
            "embedding_note": str(hp["encoder_note"]),
        }

        try:
            (run_dir / "train_ui_result.json").write_text(
                json.dumps(result, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

        from src.services.persist_service import persist_training_run

        persist_training_run(
            job_id=job_id,
            standard_src_name=display_src_name or path_left.name,
            standard_tgt_name=display_tgt_name or path_right.name,
            aligned=aligned,
            result=result,
            artifact_path=artifact_abs,
        )

        _append_hyperparam_log(job_id, hyperparams, result)

        _upd(100, "Baigta. Modelis ir DB įrašyti — galite eiti į Tikrinimą.")
        update_job(
            job_id,
            status="done",
            progress=100,
            message="Baigta.",
            result=result,
            error=None,
        )
        return result
    except Exception as e:
        update_job(
            job_id,
            status="error",
            progress=0,
            message="Klaida.",
            result=None,
            error=str(e),
        )
        return None


def get_train_job_api_fallback(job_id: str) -> dict[str, object] | None:
    """
    Kai `pair_train_job_store` nebeturi įrašo (serverio perkrova, `remove_job` ir pan.),
    atkurti /train/api atsakymą iš disko — kad progreso puslapis vėl nupieštų grafikus.
    """
    p = config.STANDARD_RUNS_FOLDER / job_id / "train_ui_result.json"
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None
    return {
        "status": "done",
        "progress": 100,
        "message": "Baigta.",
        "result": raw,
        "error": None,
    }


_TRAIN_RUN_JOB_ID_RE = re.compile(r"^[a-f0-9]{16}$")


def _load_train_ui_result_file(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return raw if isinstance(raw, dict) else None


def _normalize_active_model(mt: object) -> str:
    s = (str(mt) if mt is not None else "svm").strip().lower()
    return s if s in ("svm", "mlp") else "svm"


def _read_latest_job_id_marker() -> str | None:
    p = config.LATEST_STANDARD_RUN_FILE
    if not p.is_file():
        return None
    jid = p.read_text(encoding="utf-8").strip()
    return jid if _TRAIN_RUN_JOB_ID_RE.match(jid) else None


def _find_newest_train_ui_result_path() -> Path | None:
    root = config.STANDARD_RUNS_FOLDER
    if not root.is_dir():
        return None
    best_m: float = -1.0
    best_p: Path | None = None
    for f in root.glob("*/train_ui_result.json"):
        if not f.is_file():
            continue
        try:
            m = float(f.stat().st_mtime_ns)
        except OSError:
            continue
        if m > best_m:
            best_m = m
            best_p = f
    return best_p


def get_latest_train_ui_snapshot() -> dict[str, object] | None:
    """
    Duomenys grafikams: pirmiausia aktyvus DB įrašas, tada LATEST žymeklis,
    paskui naujausias `train_ui_result.json` po `standard_runs/` (atspariau
    seniems įrašams be JSON šalia artefakto arba kai nėra aktyvaus DB eilutės).
    """
    from src.services.persist_service import (
        get_active_training_run,
        get_training_run_by_job_id,
    )

    def pack(
        job_id: str,
        result: dict[str, object],
        src: str,
        tgt: str,
        mt: str,
    ) -> dict[str, object]:
        return {
            "job_id": job_id,
            "active_model_type": _normalize_active_model(mt),
            "result": result,
            "standard_src_name": src,
            "standard_tgt_name": tgt,
        }

    tr = get_active_training_run()
    if tr is not None:
        jid = str(tr.job_id)
        mt = getattr(tr, "active_model_type", "svm")
        src_n, tgt_n = tr.standard_src_name, tr.standard_tgt_name
        paths: list[Path] = []
        try:
            paths.append(Path(tr.artifact_path).resolve().parent / "train_ui_result.json")
        except (OSError, ValueError):
            pass
        paths.append(config.STANDARD_RUNS_FOLDER / jid / "train_ui_result.json")
        seen: set[str] = set()
        for sp in paths:
            key = str(sp)
            if key in seen:
                continue
            seen.add(key)
            data = _load_train_ui_result_file(sp)
            if data is not None:
                return pack(jid, data, src_n, tgt_n, mt)

    lid = _read_latest_job_id_marker()
    if lid:
        sp = config.STANDARD_RUNS_FOLDER / lid / "train_ui_result.json"
        data = _load_train_ui_result_file(sp)
        if data is not None:
            tr2 = get_training_run_by_job_id(lid)
            src = tr2.standard_src_name if tr2 else "—"
            tgt = tr2.standard_tgt_name if tr2 else "—"
            mt = getattr(tr2, "active_model_type", "svm") if tr2 else "svm"
            return pack(lid, data, src, tgt, mt)

    newest = _find_newest_train_ui_result_path()
    if newest is not None:
        jid = newest.parent.name
        if _TRAIN_RUN_JOB_ID_RE.match(jid):
            data = _load_train_ui_result_file(newest)
            if data is not None:
                tr3 = get_training_run_by_job_id(jid)
                src = tr3.standard_src_name if tr3 else "—"
                tgt = tr3.standard_tgt_name if tr3 else "—"
                mt = getattr(tr3, "active_model_type", "svm") if tr3 else "svm"
                return pack(jid, data, src, tgt, mt)

    return None


def load_latest_bundle() -> tuple[StandardPairBundle, str, str] | None:
    """Grąžina (bundle, job_id, active_model_type) arba None."""
    try:
        from src.services.persist_service import get_active_training_run

        tr = get_active_training_run()
        if tr is not None:
            bp = Path(tr.artifact_path)
            if bp.is_file():
                raw = joblib.load(bp)
                mt = (getattr(tr, "active_model_type", None) or "svm").strip().lower()
                if mt not in ("svm", "mlp"):
                    mt = "svm"
                return unpack_bundle_dict(raw), tr.job_id, mt
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
    return unpack_bundle_dict(raw), job_id, "svm"
