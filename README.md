# TermSync

Life Sciences reguliacinės terminologijos versijų valdymo platforma vertėjams.

## Apie projektą

TermSync sprendžia du pagrindinius Life Sciences vertėjų skausmus:

1. **Terminologijos atnaujinimas** — MedDRA žodynas atnaujinamas 2x per metus (~1000+ pakeitimų). TermSync automatiškai identifikuoja kas pasikeitė ir kur reikia atnaujinti vertimus.

2. **Dokumentų šablonų sinchronizavimas** — EMA, FDA, ICH reguliacinių dokumentų šablonai (QRD templates) keičiasi. TermSync lygina dvi šablono versijas ir parodo vertėjui tiksliai kas pasikeitė.

## Du pagrindiniai moduliai

### Modulis A — Terminologijos atnaujinimas
- Importuoja MedDRA ASC failus
- Aptinka pakeitimus tarp versijų naudojant BERT panašumo skaičiavimą
- Eksportuoja pakeitimus į TMX formatą CAT įrankiams

### Modulis B — Dokumentų šablonų sinchronizavimas
- Importuoja Word (.docx) šablonus
- Lygina dvi QRD šablono versijas segmentas po segmento
- Pažymi pasikeitusius segmentus vertėjui

## Techninė aplinka

- Python 3.12+
- Flask 3.x — web framework
- SQLAlchemy 2.x — duomenų bazė (SQLite)
- Sentence-BERT (paraphrase-multilingual-MiniLM-L12-v2) — semantinis panašumas
- SVM (sklearn) — terminų klasifikavimas
- PyTorch MPS — Apple Silicon GPU palaikymas

## Projekto struktūra

TermSync/ ├── app.py # Flask aplikacijos paleidimas ├── config.py # Konfigūracija ├── src/ │ ├── database/ # SQLAlchemy modeliai, ryšys, užklausos │ ├── parsers/ # MedDRA, TMX, DOCX parseriai │ ├── models/ # BERT ir SVM ML modeliai │ ├── services/ # Verslo logika │ └── api/ # Flask routes ├── ui/ │ ├── templates/ # HTML puslapiai (Jinja2) │ └── static/ # CSS, JS, grafikai └── experiments/ # ML eksperimentai ir grafikai


## Paleidimas

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py

Atidaryti naršyklėje: http://localhost:5000

ML eksperimentai
python experiments/run_experiments.py
python experiments/plot_results.py
Autorius
Martynas Dziugas — Baigiamasis darbas 2026