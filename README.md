# TermSync

Terminologijos ir tekstų versijų įrankis vertėjams ir lokalizacijai (tinka ir gyvybės mokslams, ir kitoms sritims).

## Apie projektą

1. **MedDRA (ar panašūs žodynai)** — importas, versijų skirtumai, „lokalizacijos gidas“: ką atnaujinti vertėjo kalbos žodyne pagal oficialių EN versijų pokyčius.

2. **Old & New** — du tos pačios kalbos failai (Word, PDF, TMX, XLIFF, Excel, …): indeksinis palyginimas, BERT slenkstis, **Show** su spalvota paraleline peržiūra.

3. **Source peržiūra** — keturi .docx (reference + source/target) su spalvinimu ir pasirenkamu MedDRA žymėjimu.

## Techninė aplinka

- Python 3.12+
- Flask 3.x — web framework
- SQLAlchemy 2.x — duomenų bazė (SQLite)
- Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`) — semantinis panašumas
- scikit-learn **RBF SVC** — standarto porų klasifikacija (embeddingų požymiai; aprašyta `src/models/svm_bundle.py`, treniruojama `standard_train_service`)
- PyTorch MLP (`src/models/mlp_classifier.py`) — greitos testinės metrikos mokyme
- PyTorch MPS — Apple Silicon GPU palaikymas

## Projekto struktūra

```
TermSync/
├── app.py                 # Flask paleidimas
├── config.py
├── src/
│   ├── database/          # SQLAlchemy ORM, ryšys
│   ├── parsers/          # MedDRA, TMX, DOCX, …
│   ├── models/           # ML: BaseMLModel, BertModel, MLPClassifier, svm_bundle (SVC + joblib bundle tipai)
│   ├── services/         # Mokymas, peržiūra, CSV žodynas, eksportas
│   └── api/              # Flask maršrutai
├── ui/
│   ├── templates/
│   └── static/
├── data/                 # DB, įkėlimai, standard_runs / bundle.joblib
└── experiments/
```

## Paleidimas

Instrukcijos skirtos **macOS (MacBook Pro, Apple Silicon)** — naudokite `python3` ir `source …/activate`.

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 app.py
```

Atidaryti naršyklėje: http://localhost:5000

## ML eksperimentai

```bash
python3 experiments/run_experiments.py
python3 experiments/plot_results.py
```

## Autorius

Martynas Dziugas — Baigiamasis darbas 2026
