"""
Vertėjo source/target .docx: kiekviena pastraipa susiejama su standarto pora (embeddingai + SVM).
Žymėjimas: tik source kalba (EN–EN tarp vertėjo source ir standarto source); difflib žymi sutapimus.
Jei EN–EN nėra jokių difflib „equal“ segmentų — visa eilutė (EN–LT–EN–LT) praleidžiama.
Target (LT–LT) stulpeliai lieka be spalvų.
"""

from __future__ import annotations

import difflib
import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import config
from src.models.bert_model import BertModel
from src.models.svm_bundle import LABEL_NO_MATCH
from src.parsers.docx_parser import DocxParser
from src.services.glossary_match_service import render_glossary_column_html
from src.services.standard_train_service import load_latest_bundle


def _cosine_dense(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pair_cosine_vectors(
    es: np.ndarray,
    et: np.ndarray,
    emb_L: np.ndarray,
    emb_R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Visų standarto porų kosinusai (vertėjo segmentas vs kiekviena pora)."""
    n = emb_L.shape[0]
    sim_src = np.array([_cosine_dense(es, emb_L[k]) for k in range(n)], dtype=np.float64)
    sim_tgt = np.array([_cosine_dense(et, emb_R[k]) for k in range(n)], dtype=np.float64)
    return sim_src, sim_tgt


def _pick_j_embedding(sim_src: np.ndarray, sim_tgt: np.ndarray) -> int:
    n = int(sim_src.shape[0])
    if n <= 0:
        return 0
    return int(
        max(
            range(n),
            key=lambda k: (float(sim_src[k]), float(sim_tgt[k])),
        )
    )


def _review_align_mode() -> str:
    m = str(getattr(config, "REVIEW_PAIR_ALIGN_MODE", "embedding")).strip().lower()
    return m if m in ("embedding", "positional") else "embedding"


def _src_top1_top2_gap(sim_src: np.ndarray) -> float:
    if sim_src.size < 2:
        return 1.0
    ss = np.sort(sim_src)
    return float(ss[-1] - ss[-2])


def _pair_meets_confidence(
    sim_s: float,
    sim_t: float,
    src_gap: float,
    *,
    align_mode: str,
) -> bool:
    """EN ir LT pakankamai artimi; embedding režime — dar ir EN 1–2 vietos spraga (nebent labai stiprus EN)."""
    if sim_s < float(config.REVIEW_PAIR_ACCEPT_MIN_SRC_COSINE):
        return False
    if sim_t < float(config.REVIEW_PAIR_ACCEPT_MIN_TGT_COSINE):
        return False
    if align_mode == "positional":
        return True
    if sim_s >= float(config.REVIEW_PAIR_STRONG_SRC_COSINE):
        return True
    return src_gap >= float(config.REVIEW_PAIR_SRC_TOP1_TOP2_MARGIN)


def _weak_positional_overflow_cell(paragraph_index: int, n_pairs: int) -> str:
    return (
        '<div class="review-no-pair" role="status">'
        f"<strong>Per daug vertėjo pastraipų.</strong> Ši eilutė #{paragraph_index + 1}, o standarto porų tik "
        f"<code>{n_pairs}</code>. Įjunkite <code>REVIEW_POSITIONAL_PAD_LAST_PAIR</code> arba sutrumpinkite vertėjo .docx."
        "</div>"
    )


def _weak_pair_standard_cell(sim_s: float, sim_t: float, src_gap: float) -> str:
    src_thr = float(config.REVIEW_PAIR_ACCEPT_MIN_SRC_COSINE)
    tgt_thr = float(config.REVIEW_PAIR_ACCEPT_MIN_TGT_COSINE)
    margin = float(config.REVIEW_PAIR_SRC_TOP1_TOP2_MARGIN)
    strong = float(config.REVIEW_PAIR_STRONG_SRC_COSINE)
    return (
        '<div class="review-no-pair" role="status">'
        "<strong>Nėra priimtinos standarto poros.</strong> "
        "Reikia: pakankamas <em>standarto EN ↔ vertėjo EN</em> ir <em>standarto LT ↔ vertėjo LT</em> kosinusas, "
        f"be to EN signalas turi būti aiškus (1-os ir 2-os vietos skirtumas ≥ <code>{margin:.2f}</code>, "
        f"nebent EN kos. ≥ <code>{strong:.2f}</code>). "
        "Išsaugotos peržiūros DB <strong>neatsinaujina</strong> — paleiskite tikrinimą iš naujo su tais pačiais failais. "
        f"<br>Dabar: EN <code>{sim_s:.2f}</code> (min. <code>{src_thr:.2f}</code>), "
        f"LT <code>{sim_t:.2f}</code> (min. <code>{tgt_thr:.2f}</code>), "
        f"EN 1–2 skirtumas <code>{src_gap:.2f}</code>."
        "</div>"
    )


def _row_color(idx: int) -> str:
    hues = [210, 135, 45, 310, 175, 285, 25, 340]
    h = hues[idx % len(hues)]
    return f"hsl({h}, 72%, 82%)"


def _wrap(color: str, text: str) -> str:
    return (
        f'<span class="hl-issue" style="background-color:{color}">'
        f"{html.escape(text)}</span>"
    )


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    sorted_r = sorted(ranges, key=lambda t: (t[0], t[1]))
    out: list[tuple[int, int]] = []
    cs, ce = sorted_r[0]
    for lo, hi in sorted_r[1:]:
        if lo <= ce:
            ce = max(ce, hi)
        else:
            out.append((cs, ce))
            cs, ce = lo, hi
    out.append((cs, ce))
    return out


def _whitespace_word_spans(s: str) -> list[tuple[int, int]]:
    """Ne tarpų segmentai [lo, hi) — „žodis“ pagal whitespace."""
    words: list[tuple[int, int]] = []
    n = len(s)
    i = 0
    while i < n:
        if s[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not s[j].isspace():
            j += 1
        words.append((i, j))
        i = j
    return words


def _ranges_highlight_full_words(s: str, raw: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Jei paryškinta bent viena žodžio raidė — žymimas visas tas žodis; nesujungiami skirtingi žodžiai."""
    if not raw:
        return []
    covered: set[int] = set()
    for lo, hi in raw:
        lo = max(0, min(lo, len(s)))
        hi = max(lo, min(hi, len(s)))
        for k in range(lo, hi):
            covered.add(k)
    picked: list[tuple[int, int]] = []
    for ws, we in _whitespace_word_spans(s):
        if ws < we and any(k in covered for k in range(ws, we)):
            picked.append((ws, we))
    return _merge_ranges(picked)


def _html_with_word_highlights(s: str, highlight_ranges: list[tuple[int, int]], color: str) -> str:
    if not highlight_ranges:
        return html.escape(s)
    ranges = _ranges_highlight_full_words(s, highlight_ranges)
    if not ranges:
        return html.escape(s)
    parts: list[str] = []
    cur = 0
    for lo, hi in ranges:
        lo = max(lo, cur)
        if lo >= hi:
            continue
        if lo > cur:
            parts.append(html.escape(s[cur:lo]))
        parts.append(_wrap(color, s[lo:hi]))
        cur = hi
    if cur < len(s):
        parts.append(html.escape(s[cur:]))
    return "".join(parts)


def _html_std_left_from_ts_L(ts: str, L: str, color: str) -> str:
    sm = difflib.SequenceMatcher(None, ts, L, autojunk=False)
    spans: list[tuple[int, int]] = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" and j1 < j2:
            spans.append((j1, j2))
    return _html_with_word_highlights(L, spans, color)


def _en_en_has_equal(ts: str, L: str) -> bool:
    """Ar vertėjo source ir standarto source turi bent vieną difflib „equal“ atkarpą."""
    sm = difflib.SequenceMatcher(None, ts, L, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" and i1 < i2 and j1 < j2:
            return True
    return False


def _html_trans_src_equal_ts_L(ts: str, L: str, color: str) -> str:
    """Vertėjo source: paryškina dalis, kurios *sutampa* su standarto source (EN–EN)."""
    sm = difflib.SequenceMatcher(None, ts, L, autojunk=False)
    spans: list[tuple[int, int]] = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal" and i1 < i2:
            spans.append((i1, i2))
    return _html_with_word_highlights(ts, spans, color)


def _plain_four(ts: str, tt: str, L: str, R: str) -> tuple[str, str, str, str]:
    return (
        html.escape(L),
        html.escape(R),
        html.escape(ts),
        html.escape(tt),
    )


@dataclass
class ReviewRow:
    std_src_html: str
    std_tgt_html: str
    trans_src_html: str
    trans_tgt_html: str
    flagged: bool
    glossary_html: str = ""


def run_translator_review(
    path_trans_src: Path,
    path_trans_tgt: Path,
    bert: BertModel | None = None,
    *,
    use_glossary: bool = True,
) -> tuple[list[ReviewRow], str]:
    loaded = load_latest_bundle()
    if loaded is None:
        raise ValueError(
            "Nėra įrašyto standarto modelio. Pirmiausia paleiskite mokymą su standarto poromis."
        )
    bundle, run_id = loaded
    svm = bundle.svm
    pairs = bundle.standard_pairs
    if not pairs:
        raise ValueError("Standarto porų rinkinys tuščias.")

    bert = bert or BertModel()
    lefts = [p[0] for p in pairs]
    rights = [p[1] for p in pairs]
    emb_L = np.array(bert.predict(lefts), dtype=np.float64)
    emb_R = np.array(bert.predict(rights), dtype=np.float64)

    pt = DocxParser(path_trans_src)
    pr = DocxParser(path_trans_tgt)
    if not pt.validate() or not pr.validate():
        raise ValueError("Vertėjo failai turi būti galiojantys .docx.")
    trans_s = pt.extract_segments()
    trans_t = pr.extract_segments()
    if not trans_s or not trans_t:
        raise ValueError("Vertėjo dokumentuose nerasta ne tuščių pastraipų.")
    n = min(len(trans_s), len(trans_t))

    glossary_entries: list[tuple[str, str, str | None]] | None = None
    if use_glossary:
        from src.services.persist_service import load_glossary_entries_for_review

        glossary_entries = load_glossary_entries_for_review()

    rows: list[ReviewRow] = []
    issue_idx = 0
    dash = '<span class="gloss-none">—</span>'
    n_pairs = len(pairs)
    align_mode = _review_align_mode()
    pad_last = bool(getattr(config, "REVIEW_POSITIONAL_PAD_LAST_PAIR", True))

    for i in range(n):
        ts, tt = trans_s[i], trans_t[i]
        es = np.array(bert.predict([ts])[0], dtype=np.float64)
        et = np.array(bert.predict([tt])[0], dtype=np.float64)

        sim_src_vec, sim_tgt_vec = _pair_cosine_vectors(es, et, emb_L, emb_R)
        if align_mode == "positional":
            if i >= n_pairs and not pad_last:
                gloss_col = (
                    render_glossary_column_html(ts, tt, glossary_entries)
                    if glossary_entries
                    else dash
                )
                rows.append(
                    ReviewRow(
                        std_src_html=_weak_positional_overflow_cell(i, n_pairs),
                        std_tgt_html='<div class="review-no-pair review-no-pair--muted">—</div>',
                        trans_src_html=html.escape(ts),
                        trans_tgt_html=html.escape(tt),
                        flagged=True,
                        glossary_html=gloss_col,
                    )
                )
                continue
            j = min(i, n_pairs - 1) if n_pairs > 0 else 0
        else:
            j = _pick_j_embedding(sim_src_vec, sim_tgt_vec)

        sim_s = float(sim_src_vec[j])
        sim_t = float(sim_tgt_vec[j])
        src_gap = _src_top1_top2_gap(sim_src_vec)
        L, R = pairs[j]

        gloss_col = (
            render_glossary_column_html(ts, tt, glossary_entries)
            if glossary_entries
            else dash
        )

        if not _pair_meets_confidence(sim_s, sim_t, src_gap, align_mode=align_mode):
            rows.append(
                ReviewRow(
                    std_src_html=_weak_pair_standard_cell(sim_s, sim_t, src_gap),
                    std_tgt_html='<div class="review-no-pair review-no-pair--muted">—</div>',
                    trans_src_html=html.escape(ts),
                    trans_tgt_html=html.escape(tt),
                    flagged=True,
                    glossary_html=gloss_col,
                )
            )
            continue

        feat = np.hstack([es, et]).reshape(1, -1)
        pred = str(svm.predict(feat)[0])

        flagged = (pred == LABEL_NO_MATCH) or (
            sim_t < float(config.REVIEW_TGT_SIM_THRESHOLD)
            and sim_s >= float(config.REVIEW_SRC_MIN_SIM)
        )

        if not flagged:
            a, b, c, d = _plain_four(ts, tt, L, R)
            rows.append(ReviewRow(a, b, c, d, False, gloss_col))
            continue

        if not _en_en_has_equal(ts, L):
            continue

        color = _row_color(issue_idx)
        h_L = _html_std_left_from_ts_L(ts, L, color)
        h_ts = _html_trans_src_equal_ts_L(ts, L, color)
        if "hl-issue" not in h_L and "hl-issue" not in h_ts:
            continue
        issue_idx += 1
        rows.append(
            ReviewRow(
                std_src_html=h_L,
                std_tgt_html=html.escape(R),
                trans_src_html=h_ts,
                trans_tgt_html=html.escape(tt),
                flagged=True,
                glossary_html=gloss_col,
            )
        )

    return rows, run_id
