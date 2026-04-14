"""
Offline SentenceTransformer fine-tuning (CosineSimilarityLoss).

Numatytasis korpusas: .docx su Word lentelėmis (du stulpeliai: EN, LT vienoje lentelėje).
Alternatyva: du atskiri .docx su poziciniu poravimu (režimas dual_docx).

Paleidimas (macOS / MacBook Pro): iš repo šaknies, su aktyviu venv, pvz.:
  python3 -m scripts.offline_st_finetune.train --output-name my_run

CPU (MacBook): numatytai ribojamos poros ir max_seq_length (RAM). Pilnas korpusas: --no-cpu-cap
Didžiausias .docx: išimkite iš data/finetune_docx/ arba --exclude-substring SMQ
Jei vis tiek OOM: mažinkite --max-pairs ir --max-seq-length
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
from pathlib import Path

import torch

# Sumažina papildomą RAM dėl tokenizer paralelumo (ypač macOS).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from torch.utils.data import DataLoader

from config import config
from sentence_transformers import InputExample, SentenceTransformer, losses
from src.parsers.docx_parser import DocxParser


def _resolve_device(name: str) -> torch.device:
    n = name.strip().lower()
    if n == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if n == "mps":
        if not torch.backends.mps.is_available():
            print("Klaida: MPS nepasiekiamas (reikia Apple Silicon ir PyTorch su MPS).", file=sys.stderr)
            sys.exit(1)
        return torch.device("mps")
    if n == "cuda":
        if not torch.cuda.is_available():
            print("Klaida: CUDA nepasiekiama.", file=sys.stderr)
            sys.exit(1)
        return torch.device("cuda")
    if n == "cpu":
        return torch.device("cpu")
    print(f"Klaida: --device turi būti auto, mps, cuda arba cpu (gauta: {name}).", file=sys.stderr)
    sys.exit(1)


def _is_probably_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return any(
        x in msg
        for x in (
            "out of memory",
            "mps backend out of memory",
            "cuda out of memory",
            "cublas error",
            "cannot allocate memory",
            "would exceed system memory",
        )
    )


def _print_oom_hints(device: torch.device, batch_size: int, output_name: str) -> None:
    base = (
        "python3 -m scripts.offline_st_finetune.train "
        f"--output-name {output_name}"
    )
    if device.type == "cpu":
        print(
            "\nKlaida: nepakanka operatyviosios atminties (RAM) CPU mokymui.\n"
            "  Uždarykite naršyklę ir kitas programas. Toliau — agresyvesnis RAM taupymas:\n",
            file=sys.stderr,
        )
        if batch_size > 1:
            for nb in (4, 2, 1):
                if batch_size > nb:
                    print(f"  {base} --device cpu --batch-size {nb}", file=sys.stderr)
            print("", file=sys.stderr)
        print(
            f"  {base} --device cpu --exclude-substring SMQ\n"
            f"  {base} --device cpu --max-seq-length 32 --max-pairs 80 --epochs 1\n"
            f"  (Arba laikinai išimkite didžiausią .docx iš data/finetune_docx/)\n",
            file=sys.stderr,
        )
        return
    if device.type == "mps":
        print(
            "\nKlaida: nepakanka MPS (Apple GPU) atminties fine-tuningui.\n",
            file=sys.stderr,
        )
        if batch_size > 1:
            print(
                "  Paskutinis bandymas ant MPS:\n"
                f"  {base} --device mps --batch-size 1\n",
                file=sys.stderr,
            )
        print(
            "  Stabiliau ant šio MacBook (RAM vietoj MPS):\n"
            f"  {base} --device cpu --batch-size 1 --max-seq-length 128\n",
            file=sys.stderr,
        )
        return
    if device.type == "cuda" and batch_size > 1:
        nb = max(1, batch_size // 2)
        print(
            "\nKlaida: nepakanka CUDA VRAM fine-tuningui.\n"
            f"  {base} --device cuda --batch-size {nb}\n"
            f"  {base} --device cpu --batch-size 1 --max-seq-length 128\n",
            file=sys.stderr,
        )
        return
    print(
        "\nKlaida: nepakanka atminties fine-tuningui.\n"
        f"  {base} --device cpu --batch-size 1 --max-seq-length 128\n",
        file=sys.stderr,
    )


def _load_pairs_bilingual_tables(
    docx_dir: Path,
    *,
    en_col_1based: int,
    lt_col_1based: int,
    skip_header: bool,
    smart_column_layout: bool,
    exclude_name_substrings: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Visi *.docx kataloge: poros iš lentelių (EN stulpelis, LT stulpelis)."""
    if en_col_1based < 1 or lt_col_1based < 1:
        print("Klaida: --en-col ir --lt-col turi būti >= 1 (Word stulpelių numeracija).", file=sys.stderr)
        sys.exit(1)
    if en_col_1based == lt_col_1based:
        print("Klaida: EN ir LT stulpeliai turi būti skirtingi.", file=sys.stderr)
        sys.exit(1)
    src_i = en_col_1based - 1
    tgt_i = lt_col_1based - 1
    raw_paths = sorted(docx_dir.glob("*.docx"))
    excl = [s.strip() for s in (exclude_name_substrings or []) if s.strip()]
    paths: list[Path] = []
    for p in raw_paths:
        name = p.name
        if excl and any(sub.lower() in name.lower() for sub in excl):
            print(f"  (--exclude-substring) praleidžiama: {name}")
            continue
        paths.append(p)
    if not paths:
        print(f"Klaida: kataloge nėra tinkamų .docx failų: {docx_dir}", file=sys.stderr)
        sys.exit(1)
    all_pairs: list[tuple[str, str]] = []
    for p in paths:
        parser = DocxParser(p)
        if not parser.validate():
            continue
        pairs = parser.extract_bilingual_table_pairs(
            source_col=src_i,
            target_col=tgt_i,
            skip_header=skip_header,
            smart_column_layout=smart_column_layout,
        )
        print(f"  {p.name}: {len(pairs)} porų iš lentelių")
        all_pairs.extend(pairs)
    return all_pairs


def _load_pairs_dual_docx(
    docx_dir: Path,
    stem: str,
    source_suffix: str,
    target_suffix: str,
) -> list[tuple[str, str]]:
    """Du failai: {stem}{source_suffix}.docx ir {stem}{target_suffix}.docx — pozicinis poravimas."""
    src_path = docx_dir / f"{stem}{source_suffix}.docx"
    tgt_path = docx_dir / f"{stem}{target_suffix}.docx"
    if not src_path.is_file():
        print(f"Klaida: nerastas šaltinio failas: {src_path}", file=sys.stderr)
        sys.exit(1)
    if not tgt_path.is_file():
        print(f"Klaida: nerastas tikslo failas: {tgt_path}", file=sys.stderr)
        sys.exit(1)
    src_segs = DocxParser(src_path).extract_segments()
    tgt_segs = DocxParser(tgt_path).extract_segments()
    n_src, n_tgt = len(src_segs), len(tgt_segs)
    if n_src != n_tgt:
        print(
            f"Klaida: poziciniam poravimui pastraipų skaičius turi sutapti. "
            f"Šaltinis {src_path.name}: {n_src}, tikslas {tgt_path.name}: {n_tgt}.",
            file=sys.stderr,
        )
        sys.exit(1)
    return list(zip(src_segs, tgt_segs, strict=True))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline SentenceTransformer fine-tuning (CosineSimilarityLoss)."
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochų skaičius (numatytai: 3 su CUDA/MPS ir --no-cpu-cap; CPU RAM taupymas: 1)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch dydis (numatytai: 4 MPS, 1 CPU, 16 CUDA). CPU: jei OOM — --max-seq-length / --max-pairs.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        metavar="NAME",
        help="auto | mps | cuda | cpu — mokymo įrenginys (MacBook: MPS dažnai OOM; stabilu --device cpu)",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="finetuned_run",
        help="Poaplankis po FINETUNED_ENCODER_OUTPUT_DIR",
    )
    p.add_argument(
        "--mode",
        choices=("bilingual_table", "dual_docx"),
        default="bilingual_table",
        help="bilingual_table: vienas .docx su EN|LT lentele; dual_docx: du .docx, pozicinis poravimas",
    )
    p.add_argument(
        "--dual-stem",
        type=str,
        default="corpus",
        help="dual_docx: failų šaknis (corpus -> corpus_en.docx + corpus_lt.docx su numatytais sufiksais)",
    )
    p.add_argument(
        "--source-suffix",
        type=str,
        default="_en",
        help="dual_docx: šaltinio failo sufiksas prieš .docx",
    )
    p.add_argument(
        "--target-suffix",
        type=str,
        default="_lt",
        help="dual_docx: tikslo failo sufiksas prieš .docx",
    )
    p.add_argument(
        "--en-col",
        type=int,
        default=1,
        help="bilingual_table: EN stulpelis (1 = pirmas Word stulpelis)",
    )
    p.add_argument(
        "--lt-col",
        type=int,
        default=2,
        help="bilingual_table: LT stulpelis (1 = pirmas Word stulpelis)",
    )
    p.add_argument(
        "--no-skip-header",
        action="store_true",
        help="bilingual_table: nepraleisti antraštės eilutės net jei atrodo kaip CAT antraštė",
    )
    p.add_argument(
        "--no-smart-layout",
        action="store_true",
        help=(
            "bilingual_table: išjungti automatinį Source(en)/Target(lt) aptikimą ir "
            "stulpelių pernaudojimą tarp lentelių — naudojami tik --en-col / --lt-col "
            "(pvz. MemoQ be šio flag: --en-col 4 --lt-col 6)"
        ),
    )
    p.add_argument("--lr", type=float, default=2e-5, help="AdamW learning rate")
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        metavar="N",
        help="Tokenų limitas porai (mažiau RAM), pvz. 96 arba 128",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        metavar="N",
        help="Atsitiktinai mokyti tik N porų (mažiau RAM / greitas bandymas)",
    )
    p.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="CPU: neįjungti gradient checkpointing (numatytai CPU įjungta — mažiau RAM)",
    )
    p.add_argument(
        "--no-cpu-cap",
        action="store_true",
        help=(
            "CPU: neautomatiškai neriboti max_seq_length ir porų skaičiaus "
            "(numatytai taupoma RAM; pilnam korpusui ant stipraus PC naudokite šį flagą)"
        ),
    )
    p.add_argument(
        "--exclude-substring",
        action="append",
        default=None,
        metavar="TEXT",
        help=(
            "bilingual_table: praleisti .docx, jei faile yra ši eilutė vardu (kartoti flagą). "
            "Pvz. didžiausią korpusą: --exclude-substring SMQ"
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Išjungti progress bar",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    docx_dir = Path(config.FINETUNE_DOCX_DIR)
    if not docx_dir.is_dir():
        print(f"Klaida: katalogas neegzistuoja: {docx_dir}", file=sys.stderr)
        sys.exit(1)

    skip_header = not args.no_skip_header
    smart_layout = not args.no_smart_layout
    if args.mode == "bilingual_table":
        layout_note = (
            "išmanusis CAT stulpelių išdėstymas (Source (en) / Target (lt))"
            if smart_layout
            else f"fiksuoti stulpeliai EN={args.en_col}, LT={args.lt_col}"
        )
        print(f"Kraunamos poros (lentelės, {layout_note}) iš {docx_dir}:")
        pairs = _load_pairs_bilingual_tables(
            docx_dir,
            en_col_1based=args.en_col,
            lt_col_1based=args.lt_col,
            skip_header=skip_header,
            smart_column_layout=smart_layout,
            exclude_name_substrings=list(args.exclude_substring)
            if args.exclude_substring
            else [],
        )
    else:
        print(
            f"Kraunamos poros (dual_docx, šaknis «{args.dual_stem}», "
            f"{args.source_suffix} / {args.target_suffix}) iš {docx_dir}:"
        )
        pairs = _load_pairs_dual_docx(
            docx_dir,
            stem=args.dual_stem,
            source_suffix=args.source_suffix,
            target_suffix=args.target_suffix,
        )
        print(f"  Porų skaičius: {len(pairs)}")

    n_pairs_loaded = len(pairs)
    if args.max_pairs is not None:
        if args.max_pairs < 1:
            print("Klaida: --max-pairs turi būti >= 1.", file=sys.stderr)
            sys.exit(1)
        k = min(args.max_pairs, len(pairs))
        if k < len(pairs):
            rng = random.Random(42)
            pairs = rng.sample(pairs, k)
            print(f"  --max-pairs: naudojama {k} porų (iš {n_pairs_loaded})")
        else:
            print(f"  --max-pairs: naudojamos visos {len(pairs)} poros")

    device = _resolve_device(args.device)
    # CPU: numatytai mažiau porų = mažiau RAM (MacBook su 8–16 GB).
    _cpu_pair_cap = 100
    _n_before_cpu_cap = len(pairs)
    if (
        device.type == "cpu"
        and not args.no_cpu_cap
        and args.max_pairs is None
        and len(pairs) > _cpu_pair_cap
    ):
        rng = random.Random(42)
        pairs = rng.sample(pairs, _cpu_pair_cap)
        print(
            f"  CPU: numatytai {_cpu_pair_cap} atsitiktinių porų iš {_n_before_cpu_cap} "
            "(RAM taupymas). Visam korpusui: --no-cpu-cap arba --max-pairs <skaičius>"
        )

    if not pairs:
        print(
            "Klaida: nerasta nei vienos poros. bilingual_table: ar .docx turi lenteles su EN/LT? "
            "MemoQ eksporte įjunkite numatytąjį išmanųjį režimą (be --no-smart-layout). "
            "Paprastai du stulpeliai: --no-smart-layout ir --en-col 1 --lt-col 2. "
            "Jei pirmoji eilutė jau yra vertimas (ne antraštė): --no-skip-header.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.epochs is not None and args.epochs < 1:
        print("Klaida: --epochs turi būti >= 1.", file=sys.stderr)
        sys.exit(1)
    if args.epochs is None:
        training_epochs = 1 if device.type == "cpu" and not args.no_cpu_cap else 3
    else:
        training_epochs = args.epochs

    if device.type == "cpu" and not args.no_cpu_cap:
        _max_chars = 256
        pairs = [(s[:_max_chars], t[:_max_chars]) for s, t in pairs]
        print(
            f"  CPU: tekstai trumpinami iki {_max_chars} simb. kiekvienoje poroje "
            "(RAM). Pilni segmentai: --no-cpu-cap"
        )

    out_dir = (Path(config.FINETUNED_ENCODER_OUTPUT_DIR) / args.output_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    if batch_size is None:
        if device.type == "mps":
            batch_size = 4
        elif device.type == "cpu":
            batch_size = 1
        else:
            batch_size = 16
        print(
            f"Įrenginys: {device} (numatytasis batch_size={batch_size}; "
            f"MPS dažnai OOM → --device cpu; CPU: --max-seq-length, --max-pairs jei vis tiek OOM)",
        )
    else:
        print(f"Įrenginys: {device}")
    msl_note = f", max_seq_length={args.max_seq_length}" if args.max_seq_length else ""
    print(
        f"Mokymas: {len(pairs)} porų, epochs={training_epochs}, "
        f"batch_size={batch_size}, lr={args.lr}{msl_note}"
    )

    if device.type == "cpu":
        try:
            if args.no_cpu_cap:
                nt = min(4, max(1, (os.cpu_count() or 4) // 2))
            else:
                nt = 1
            torch.set_num_threads(nt)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    train_examples = [
        InputExample(texts=[s, t], label=1.0) for s, t in pairs
    ]
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
    )

    model = SentenceTransformer(
        config.BERT_MODEL_NAME,
        device=str(device),
    )
    _orig_max_seq = int(model.max_seq_length)
    if args.max_seq_length is not None:
        if args.max_seq_length < 8 or args.max_seq_length > 512:
            print("Klaida: --max-seq-length turi būti tarp 8 ir 512.", file=sys.stderr)
            sys.exit(1)
        model.max_seq_length = args.max_seq_length
        print(f"  max_seq_length nustatytas į {model.max_seq_length}")
    elif device.type == "cpu" and not args.no_cpu_cap:
        _cpu_msl = min(40, _orig_max_seq)
        model.max_seq_length = _cpu_msl
        print(
            f"  CPU: max_seq_length={model.max_seq_length} (numatytas RAM taupymas, "
            f"modelis leidžia iki {_orig_max_seq}). Pilnas ilgis: --no-cpu-cap arba --max-seq-length {_orig_max_seq}"
        )

    if device.type == "cpu" and not args.no_gradient_checkpointing:
        try:
            tr = model[0]
            am = getattr(tr, "auto_model", None)
            if am is not None and hasattr(am, "gradient_checkpointing_enable"):
                am.gradient_checkpointing_enable()
                print("  Gradient checkpointing: įjungta (CPU, mažiau RAM)")
        except (AttributeError, IndexError, TypeError):
            pass

    train_loss = losses.CosineSimilarityLoss(model)

    steps_per_epoch = max(1, (len(train_examples) + batch_size - 1) // batch_size)
    total_steps = max(1, steps_per_epoch * training_epochs)
    warmup_steps = max(10, int(0.1 * total_steps))
    warmup_steps = min(warmup_steps, max(5, total_steps // 4))

    gc.collect()

    _fit_kw = dict(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=training_epochs,
        warmup_steps=warmup_steps,
        output_path=str(out_dir),
        show_progress_bar=not args.quiet,
    )
    try:
        try:
            model.fit(optimizer_params={"lr": args.lr, "foreach": False}, **_fit_kw)
        except TypeError:
            model.fit(optimizer_params={"lr": args.lr}, **_fit_kw)
    except (RuntimeError, MemoryError) as e:
        if _is_probably_oom(e):
            _print_oom_hints(device, batch_size, args.output_name)
            sys.exit(1)
        raise

    abs_out = str(out_dir.resolve())
    print()
    print(f"Išsaugota: {abs_out}")
    print(
        "Prieš `python3 app.py` nustatykite bazinį encoderį į šį katalogą, pvz. aplinkoje: "
        "export TERMSYNC_BERT_MODEL_NAME="
        f'"{abs_out}"'
        " (arba config lauką TERMSYNC_BERT_MODEL_NAME)."
    )


if __name__ == "__main__":
    main()
