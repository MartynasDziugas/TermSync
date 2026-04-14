"""
Čia keičiate offline fine-tuning numatytuosius parametrus (be ilgų terminalo vėliavų).

Paleidimas lieka:
  python3 -m scripts.offline_st_finetune.train

Numatytasis profilis žemiau sutampa su `train.py` MacBook / RAM logika, aprašyta projekte:
  --device cpu, be --no-cpu-cap → automatiškai: iki 100 porų, max_seq_length 40,
  256 simb./pora, 1 epocha, batch 1, gradient checkpointing (žr. train.py).

Pilnas korpusas / ilgesnės sekos / daugiau epochų: tik jei sąmoningai rizikuojate RAM
(`NO_CPU_CAP = True` ir/ar rankiniai MAX_* / EPOCHS) arba stipresnė mašina.

Terminalo argumentai vis tiek gali perrašyti šias reikšmes.
"""

from __future__ import annotations

# --- Pagrindinė (nelūžimo numatytieji = leisti train.py CPU „cap“) ---
# auto | cpu | mps | cuda  — MacBook: stabilu „cpu“
DEVICE: str = "cpu"

# Išvesties poaplankis po data/finetuned_encoder/
OUTPUT_NAME: str = "ft_mac_safe"

# None = train.py: CPU be --no-cpu-cap → 1 epocha; su --no-cpu-cap → 3 (arba MPS/CUDA)
EPOCHS: int | None = None

# None = train.py: CPU → 1; MPS → 4; CUDA → 16
BATCH_SIZE: int | None = None

# Mokymo greitis (AdamW)
LR: float = 2e-5

# bilingual_table | dual_docx
MODE: str = "bilingual_table"

# --- bilingual_table (Word lentelės) ---
EN_COL: int = 1
LT_COL: int = 2

NO_SMART_LAYOUT: bool = False
NO_SKIP_HEADER: bool = False

# Dideli / rizikingi failai: praleisti vardu (pvz. SMQ sąrašai)
EXCLUDE_FILENAME_SUBSTRINGS: list[str] = ["SMQ"]

# None + NO_CPU_CAP False → train.py nustato max_seq_length=40 ant CPU
MAX_SEQ_LENGTH: int | None = None

# None + NO_CPU_CAP False → train.py imsi iki 100 atsitiktinių porų ant CPU
MAX_PAIRS: int | None = None

# False = įjungtas automatinis RAM „cap“ (rekomenduojama MacBook).
# True = pilnas korpusas / mažiau ribojimų — tik jei turite RAM atsargą.
NO_CPU_CAP: bool = False

# False = gradient checkpointing ant CPU (numatytai train.py)
NO_GRADIENT_CHECKPOINTING: bool = False

# dual_docx ---
DUAL_STEM: str = "corpus"
SOURCE_SUFFIX: str = "_en"
TARGET_SUFFIX: str = "_lt"

QUIET: bool = False


def apply_argparse_defaults(parser) -> None:
    """ArgumentParser.set_defaults pagal šio failo konstantas."""
    d: dict = {
        "device": DEVICE,
        "output_name": OUTPUT_NAME,
        "lr": LR,
        "mode": MODE,
        "en_col": EN_COL,
        "lt_col": LT_COL,
        "no_smart_layout": NO_SMART_LAYOUT,
        "no_skip_header": NO_SKIP_HEADER,
        "no_cpu_cap": NO_CPU_CAP,
        "no_gradient_checkpointing": NO_GRADIENT_CHECKPOINTING,
        "dual_stem": DUAL_STEM,
        "source_suffix": SOURCE_SUFFIX,
        "target_suffix": TARGET_SUFFIX,
        "quiet": QUIET,
    }
    if EPOCHS is not None:
        d["epochs"] = EPOCHS
    if BATCH_SIZE is not None:
        d["batch_size"] = BATCH_SIZE
    if MAX_SEQ_LENGTH is not None:
        d["max_seq_length"] = MAX_SEQ_LENGTH
    if MAX_PAIRS is not None:
        d["max_pairs"] = MAX_PAIRS
    parser.set_defaults(**d)


def default_exclude_substrings() -> list[str]:
    return list(EXCLUDE_FILENAME_SUBSTRINGS)
