"""
Sklearn RBF SVM porų klasifikatoriui + metaduomenys (standarto poros, požymių dimensija).

Saugoma `bundle.joblib` kaip dict raktai `svm`, `standard_pairs`, `feature_dim`
(suderinama su jau įrašytais artefaktais).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sklearn.svm import SVC

if TYPE_CHECKING:
    from src.models.mlp_classifier import MLPClassifier

LABEL_MATCH = "match"
LABEL_NO_MATCH = "no_match"


def new_svm_pair_classifier(
    *,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str | float = "scale",
) -> SVC:
    """SVM porų klasifikatoriui; tas pats hiperparametrų rinkinys mokymui ir pilnai imčiai."""
    k = (kernel or "rbf").lower()
    if k not in ("rbf", "linear", "poly", "sigmoid"):
        k = "rbf"
    return SVC(kernel=k, C=float(C), gamma=gamma, probability=False)


@dataclass(frozen=True)
class StandardPairBundle:
    svm: SVC
    standard_pairs: list[tuple[str, str]]
    feature_dim: int
    mlp: MLPClassifier | None = None


def pack_bundle(
    svm: SVC,
    standard_pairs: list[tuple[str, str]],
    feature_dim: int,
    mlp: MLPClassifier | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "svm": svm,
        "standard_pairs": standard_pairs,
        "feature_dim": int(feature_dim),
    }
    if mlp is not None:
        out["mlp"] = mlp
    return out


def unpack_bundle_dict(raw: Mapping[str, Any]) -> StandardPairBundle:
    if not isinstance(raw, Mapping):
        raise TypeError("bundle turi būti mapping (pvz. joblib įkeltas dict).")
    try:
        svm = raw["svm"]
        pairs_raw = raw["standard_pairs"]
    except KeyError as e:
        raise ValueError(f"netinkamas bundle: trūksta rakto {e!s}.") from e
    if not isinstance(svm, SVC):
        raise TypeError(f"laukta sklearn.svm.SVC, gauta {type(svm).__name__}.")
    pairs: list[tuple[str, str]] = [(str(a), str(b)) for a, b in pairs_raw]
    fd = raw.get("feature_dim")
    if fd is None:
        inf = getattr(svm, "n_features_in_", None)
        fd = int(inf) if inf is not None else 0
    else:
        fd = int(fd)
    mlp_raw = raw.get("mlp")
    mlp_out: MLPClassifier | None = None
    if mlp_raw is not None:
        from src.models.mlp_classifier import MLPClassifier

        if not isinstance(mlp_raw, MLPClassifier):
            raise TypeError(
                f"laukta MLPClassifier arba None, gauta {type(mlp_raw).__name__}."
            )
        mlp_out = mlp_raw
    return StandardPairBundle(
        svm=svm,
        standard_pairs=pairs,
        feature_dim=fd,
        mlp=mlp_out,
    )
