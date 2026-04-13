"""
Sklearn RBF SVM porų klasifikatoriui + metaduomenys (standarto poros, požymių dimensija).

Saugoma `bundle.joblib` kaip dict raktai `svm`, `standard_pairs`, `feature_dim`
(suderinama su jau įrašytais artefaktais).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sklearn.svm import SVC

LABEL_MATCH = "match"
LABEL_NO_MATCH = "no_match"


def new_svm_pair_classifier() -> SVC:
    """Tas pats hiperparametrų rinkinys mokymui ir pilnai imčiai."""
    return SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)


@dataclass(frozen=True)
class StandardPairBundle:
    svm: SVC
    standard_pairs: list[tuple[str, str]]
    feature_dim: int


def pack_bundle(
    svm: SVC,
    standard_pairs: list[tuple[str, str]],
    feature_dim: int,
) -> dict[str, Any]:
    return {
        "svm": svm,
        "standard_pairs": standard_pairs,
        "feature_dim": int(feature_dim),
    }


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
    return StandardPairBundle(svm=svm, standard_pairs=pairs, feature_dim=fd)
