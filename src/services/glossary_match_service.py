"""CSV žodyno atitiktys vertėjo peržiūroje: NFC + casefold, ribotas Levenshteino atstumas, nepersidengiantys hitai."""

from __future__ import annotations

import html
import unicodedata
from dataclasses import dataclass

from config import config


@dataclass(frozen=True)
class _GlossHit:
    start: int
    end: int
    expected_target: str
    target_ok: bool
    note: str | None


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _gloss_cmp(s: str) -> str:
    """Vieningas palyginimas: Unicode NFC + casefold (įvairios kalbos, ne ASCII-only lower)."""
    return _nfc(s).casefold()


def _lev_bounded(a: str, b: str, max_dist: int) -> int | None:
    """
    Levenshteino atstumas tarp a ir b; grąžina int jei <= max_dist, kitaip None (nutraukiama anksčiau).
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return None
    if max_dist == 0:
        return 0 if a == b else None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i]
        row_min = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(v)
            row_min = min(row_min, v)
        if row_min > max_dist:
            return None
        prev = cur
    d = prev[lb]
    return d if d <= max_dist else None


def _find_fuzzy_span(ts_nfc: str, st: str, max_dist: int) -> tuple[int, int] | None:
    """
    Randa atkarpą ts_nfc[start:end], kur _gloss_cmp atstumas nuo _gloss_cmp(st) <= max_dist.
    Renkama: mažiausias atstumas, tada ilgis arčiausiai termino ilgio, tada mažiausias start (kairėn).
    """
    stn = st.strip()
    if not stn:
        return None
    key = _gloss_cmp(stn)
    n = len(ts_nfc)
    ln0 = len(stn)
    best: tuple[int, int, int, int] | None = None  # dist, abs(L-ln0), start, end

    for i in range(n):
        lo = max(1, ln0 - max_dist)
        hi = min(n - i, ln0 + max_dist)
        for L in sorted(range(lo, hi + 1), key=lambda x: (abs(x - ln0), x)):
            sub = ts_nfc[i : i + L]
            d = _lev_bounded(_gloss_cmp(sub), key, max_dist)
            if d is None:
                continue
            cand = (d, abs(L - ln0), i, i + L)
            if best is None or cand < best:
                best = cand
    if best is None:
        return None
    return best[2], best[3]


def _fuzzy_expected_in_target(tt_nfc: str, expected_tgt: str, max_dist: int) -> bool:
    """Ar target pastraipoje yra segmentas, iki max_dist paklaidos atitinkantis lauktą vertimą."""
    et = expected_tgt.strip()
    if not et:
        return True
    key = _gloss_cmp(et)
    if key in _gloss_cmp(tt_nfc):
        return True
    ln0 = len(et)
    n = len(tt_nfc)
    for i in range(n):
        lo = max(1, ln0 - max_dist)
        hi = min(n - i, ln0 + max_dist)
        for L in sorted(range(lo, hi + 1), key=lambda x: (abs(x - ln0), x)):
            sub = tt_nfc[i : i + L]
            if _lev_bounded(_gloss_cmp(sub), key, max_dist) is not None:
                return True
    return False


def _interval_overlaps(s: int, e: int, used: list[tuple[int, int]]) -> bool:
    for a, b in used:
        if not (e <= a or s >= b):
            return True
    return False


def collect_glossary_hits(
    ts: str,
    tt: str,
    entries: list[tuple[str, str, str | None]],
) -> list[_GlossHit]:
    max_dist = int(getattr(config, "GLOSSARY_MAX_EDIT_DISTANCE", 3))
    ts_nfc = _nfc(ts)
    tt_nfc = _nfc(tt)
    used: list[tuple[int, int]] = []
    hits: list[_GlossHit] = []
    min_len = int(config.GLOSSARY_REVIEW_MIN_SOURCE_LEN)

    for src_term, exp_tgt, note in entries:
        st = src_term.strip()
        et = exp_tgt.strip()
        if len(st) < min_len or not et:
            continue
        span = _find_fuzzy_span(ts_nfc, st, max_dist)
        if span is None:
            continue
        s, e = span
        if _interval_overlaps(s, e, used):
            continue
        used.append((s, e))
        ok = _fuzzy_expected_in_target(tt_nfc, et, max_dist)
        hits.append(
            _GlossHit(
                start=s,
                end=e,
                expected_target=et,
                target_ok=ok,
                note=note,
            )
        )
    return hits


def _snippet(ts: str, start: int, end: int, radius: int = 28) -> str:
    lo = max(0, start - radius)
    hi = min(len(ts), end + radius)
    left_el = "…" if lo > 0 else ""
    right_el = "…" if hi < len(ts) else ""
    before = html.escape(ts[lo:start])
    mid = f'<span class="hl-gloss-snippet-term">{html.escape(ts[start:end])}</span>'
    after = html.escape(ts[end:hi])
    return f"{left_el}{before}{mid}{after}{right_el}"


def render_glossary_column_html(
    ts: str,
    tt: str,
    entries: list[tuple[str, str, str | None]] | None,
    *,
    hits: list[_GlossHit] | None = None,
) -> str:
    ts_nfc = _nfc(ts)
    tt_nfc = _nfc(tt)
    if hits is None:
        if not entries:
            return '<span class="gloss-none">—</span>'
        hits = collect_glossary_hits(ts_nfc, tt_nfc, entries)
    if not hits:
        return '<span class="gloss-none">—</span>'
    parts: list[str] = []
    for h in hits:
        snip = _snippet(ts_nfc, h.start, h.end)
        note_html = ""
        if h.note:
            note_html = f' <span class="gloss-note">{html.escape(h.note)}</span>'
        if h.target_ok:
            parts.append(
                f'<div class="gloss-line gloss-ok"><span class="gloss-mark">✓</span> {snip}'
                f'<div class="gloss-meta">CSV → <code>{html.escape(h.expected_target)}</code>'
                f"{note_html}</div></div>"
            )
        else:
            parts.append(
                f'<div class="gloss-line gloss-bad"><span class="gloss-mark">✗</span> {snip}'
                f'<div class="gloss-meta">Laukta <code>{html.escape(h.expected_target)}</code>'
                f" <em>— nerasta target faile (Levenshteinas ≤ {int(getattr(config, 'GLOSSARY_MAX_EDIT_DISTANCE', 3))})</em>{note_html}</div></div>"
            )
    return "".join(parts)
