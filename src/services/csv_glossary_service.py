"""
CSV terminų lentelė → normalizuotos eilutės (source, target, pastaba).
Antraštės: source/target, en/lt, source_term/target_term ir pan.; kitaip — pirmi du stulpeliai.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import IO

MAX_ROWS = 10_000

_SOURCE_ALIASES = frozenset(
    {
        "source",
        "source_term",
        "source_text",
        "en",
        "src",
        "šaltinis",
        "saltinis",
        "terminas_source",
    }
)
_TARGET_ALIASES = frozenset(
    {
        "target",
        "target_term",
        "target_text",
        "lt",
        "de",
        "fr",
        "tgt",
        "vertimas",
        "terminas_target",
    }
)
_NOTE_ALIASES = frozenset(
    {"note", "notes", "comment", "pastaba", "kontekstas", "context"}
)


def _norm_header(h: str) -> str:
    return re.sub(r"\s+", "", h.strip().lower())


def _pick_columns(headers: list[str]) -> tuple[int | None, int | None, int | None]:
    """Grąžina (i_src, i_tgt, i_note) arba abu None jei naudosime pirmus du stulpelius."""
    norm = [_norm_header(h) for h in headers]
    i_src: int | None = None
    i_tgt: int | None = None
    i_note: int | None = None
    for i, h in enumerate(norm):
        if h in _SOURCE_ALIASES:
            i_src = i
        elif h in _TARGET_ALIASES:
            i_tgt = i
        elif h in _NOTE_ALIASES:
            i_note = i
    if i_src is not None and i_tgt is not None:
        return i_src, i_tgt, i_note
    return None, None, None


def parse_glossary_csv(
    file_obj: IO[str],
    *,
    delimiter: str | None = None,
    has_header: bool = True,
) -> tuple[list[tuple[str, str, str | None]], list[str]]:
    """
    Grąžina (eilutės kaip (source, target, note), įspėjimai).
    Tuščios eilutės praleidžiamos.
    """
    warnings: list[str] = []
    sample = file_obj.read(8192)
    file_obj.seek(0)
    if delimiter is None:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

    reader = csv.reader(file_obj, delimiter=delimiter)
    rows_out: list[tuple[str, str, str | None]] = []

    if not has_header:
        for _idx, parts in enumerate(reader):
            if len(rows_out) >= MAX_ROWS:
                warnings.append(f"Importuota tik pirmos {MAX_ROWS} eilučių (riba).")
                break
            if len(parts) < 2:
                continue
            src = (parts[0] or "").strip()
            tgt = (parts[1] or "").strip()
            note_raw = (parts[2] or "").strip() if len(parts) > 2 else ""
            note: str | None = note_raw or None
            if not src and not tgt:
                continue
            rows_out.append((src, tgt, note))
        if not rows_out:
            raise ValueError("Nėra tinkamų eilučių (reikia bent dviejų stulpelių).")
        warnings.append(
            "Be antraštės: 1 stulpelis = source, 2 = target, 3 = pastaba (nebūtina)."
        )
        return rows_out, warnings

    try:
        header_row = next(reader)
    except StopIteration:
        return [], ["Failas tuščias."]

    headers = [h.strip() for h in header_row]
    i_src, i_tgt, i_note = _pick_columns(headers)

    if i_src is None or i_tgt is None:
        if len(headers) < 2:
            raise ValueError(
                "CSV turi turėti bent du stulpelius arba antraštėse "
                "„source“ ir „target“ (arba en / lt)."
            )
        i_src, i_tgt = 0, 1
        i_note = 2 if len(headers) > 2 else None
        warnings.append(
            "Antraštės neatpažintos — naudojami pirmi du stulpeliai kaip source / target."
        )

    for idx, parts in enumerate(reader):
        if len(rows_out) >= MAX_ROWS:
            warnings.append(f"Importuota tik pirmos {MAX_ROWS} eilučių (riba).")
            break
        if len(parts) <= max(i_src, i_tgt):
            continue
        src = (parts[i_src] or "").strip()
        tgt = (parts[i_tgt] or "").strip()
        note: str | None = None
        if i_note is not None and len(parts) > i_note:
            n = (parts[i_note] or "").strip()
            note = n or None
        if not src and not tgt:
            continue
        rows_out.append((src, tgt, note))

    if not rows_out:
        raise ValueError("Po apdorojimo neliko tinkamų eilučių (tikrinkite stulpelius ir koduotę UTF-8).")

    return rows_out, warnings


def parse_glossary_csv_path(
    path: Path,
    encoding: str = "utf-8-sig",
    *,
    has_header: bool = True,
) -> tuple[list[tuple[str, str, str | None]], list[str]]:
    text = path.read_text(encoding=encoding)
    from io import StringIO

    return parse_glossary_csv(StringIO(text), has_header=has_header)
