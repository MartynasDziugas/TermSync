# Changelog

## [1.0.0] - 2026-04-12

### Pridėta
- MedDRA ASC failo importavimas ir versijų valdymas
- QRD dokumentų šablonų importavimas (.docx)
- Dviejų QRD šablonų versijų palyginimas naudojant BERT
- Pasikeikusių segmentų eksportavimas į TMX ir DOCX
- SVM terminų klasifikatorius su BERT embeddings
- 25 hiperparametrų eksperimentai (kernel, C, gamma)
- 5 rezultatų vizualizacijos grafikai
- Flask web UI (9 puslapiai)
- SQLAlchemy ORM duomenų bazė (6 lentelės)
- Eksperimentų saugojimas į DB ir CSV

### Techninė aplinka
- Python 3.12+
- Flask 3.x
- SQLAlchemy 2.x
- Sentence-BERT (paraphrase-multilingual-MiniLM-L12-v2)
- PyTorch MPS (Apple Silicon)
- scikit-learn SVM

### Žinomi apribojimai
- TMX eksportas rodo tik angliškas segmentų poras (senas/naujas) — vertėjas turi pats išversti pasikeitusius segmentus
- IATE terminologijos integravimas nėra įgyvendintas