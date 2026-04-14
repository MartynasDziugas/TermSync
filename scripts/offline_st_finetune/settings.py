"""
Čia keičiate offline fine-tuning numatytuosius parametrus (be ilgų terminalo vėliavų).

Paleidimas lieka:
  python3 -m scripts.offline_st_finetune.train

Terminalo argumentai vis tiek gali perrašyti šias reikšmes, jei juos nurodote
(pvz. `python3 -m ... --epochs 5` perrašo EPOCHS žemiau).
"""

from __future__ import annotations

# --- Pagrindinė ---
# auto | cpu | mps | cuda
DEVICE: str = "cpu"

# Išvesties poaplankis po data/finetuned_encoder/
OUTPUT_NAME: str = "mano_run"

# None = leisti train.py logikai (CPU be --no-cpu-cap: 1, kitaip 3); arba fiksuotas int
EPOCHS: int | None = None

# None = leisti train.py (MPS:4, CPU:1, CUDA:16); arba fiksuotas int
BATCH_SIZE: int | None = None

# Mokymo greitis (AdamW)
LR: float = 2e-5

# bilingual_table | dual_docx
MODE: str = "bilingual_table"

# --- bilingual_table (Word lentelės) ---
# Stulpelių numeracija Word (1-based), naudojama tik su NO_SMART_LAYOUT
EN_COL: int = 1
LT_COL: int = 2

# Jei True: išjungiamas CAT Source/Target stulpelių automatinis aptikimas
NO_SMART_LAYOUT: bool = False

# Jei True: antraštės eilutė nepraleidžiama net jei atrodo kaip CAT antraštė
NO_SKIP_HEADER: bool = False

# Failų vardai: jei eilutė yra faile (case-insensitive), failas neįtraukiamas.
# Tuščias sąrašas [] = visi .docx iš katalogo
EXCLUDE_FILENAME_SUBSTRINGS: list[str] = ["SMQ"]

# None = train.py logika; int = fiksuotas tokenų limitas (mažiau RAM)
MAX_SEQ_LENGTH: int | None = None

# None = train.py logika; int = atsitiktinai tik N porų
MAX_PAIRS: int | None = None

# CPU: išjungti automatinį RAM ribojimą (poros, max_seq, simboliai, epochos)
NO_CPU_CAP: bool = False

# CPU: neįjungti gradient checkpointing
NO_GRADIENT_CHECKPOINTING: bool = False

# dual_docx režimui ---
DUAL_STEM: str = "corpus"
SOURCE_SUFFIX: str = "_en"
TARGET_SUFFIX: str = "_lt"

# Kita ---
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
