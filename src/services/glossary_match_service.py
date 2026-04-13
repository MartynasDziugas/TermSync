"""CSV žodyno atitiktys vertėjo peržiūroje: ilgesni source terminai pirmi, nepersidengiantys."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from config import config


@dataclass(frozen=True)
class _GlossHit:
    start: int
    end: int
    expected_target: str
    target_ok: bool
    note: str | None


def _norm_substring(s: str) -> str:
    t = s.casefold()
    return re.sub(r"\s+", " ", t).strip()


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
    used: list[tuple[int, int]] = []
    hits: list[_GlossHit] = []
    tt_norm = _norm_substring(tt)
    min_len = int(config.GLOSSARY_REVIEW_MIN_SOURCE_LEN)
    for src_term, exp_tgt, note in entries:
        st = src_term.strip()
        et = exp_tgt.strip()
        if len(st) < min_len or not et:
            continue
        try:
            pat = re.compile(re.escape(st), re.IGNORECASE)
        except re.error:
            continue
        for m in pat.finditer(ts):
            s, e = m.span()
            if _interval_overlaps(s, e, used):
                continue
            used.append((s, e))
            ok = _norm_substring(et) in tt_norm
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
    mid = f'<span class="hl-gloss">{html.escape(ts[start:end])}</span>'
    after = html.escape(ts[end:hi])
    return f"{left_el}{before}{mid}{after}{right_el}"


def render_glossary_column_html(
    ts: str,
    tt: str,
    entries: list[tuple[str, str, str | None]],
) -> str:
    hits = collect_glossary_hits(ts, tt, entries)
    if not hits:
        return '<span class="gloss-none">—</span>'
    parts: list[str] = []
    for h in hits:
        snip = _snippet(ts, h.start, h.end)
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
                f" <em>— nerasta target faile</em>{note_html}</div></div>"
            )
    return "".join(parts)
