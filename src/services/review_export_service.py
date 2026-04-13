"""Peržiūros sesijos eksportas į CSV (grynasis tekstas be HTML žymų)."""

from __future__ import annotations

import csv
import io
import re

from src.services.translator_review_service import ReviewRow

_TAG_RE = re.compile(r"<[^>]+>")


def _plain(html: str) -> str:
    t = _TAG_RE.sub("", html)
    return re.sub(r"\s+", " ", t).strip()


def review_rows_to_csv_bytes(rows: list[ReviewRow]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "row_index",
            "flagged",
            "standard_source",
            "standard_target",
            "translator_source",
            "translator_target",
            "glossary_csv",
        ]
    )
    for i, r in enumerate(rows):
        w.writerow(
            [
                i + 1,
                r.flagged,
                _plain(r.std_src_html),
                _plain(r.std_tgt_html),
                _plain(r.trans_src_html),
                _plain(r.trans_tgt_html),
                _plain(r.glossary_html),
            ]
        )
    return buf.getvalue().encode("utf-8-sig")
