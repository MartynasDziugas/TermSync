import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from src.database.connection import get_session, init_db
from src.services.experiment_service import get_all_experiments

init_db()
CHARTS_DIR = Path("ui/static/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

with get_session() as session:
    experiments = get_all_experiments(session)

names = [e.experiment_name for e in experiments]
accuracies = [e.metrics.get("accuracy_mean", 0) for e in experiments]
stds = [e.metrics.get("accuracy_std", 0) for e in experiments]
kernels = [e.hyperparameters.get("kernel", "") for e in experiments]
c_values = [e.hyperparameters.get("C", 1.0) for e in experiments]

# 1. Tikslumo palyginimas per eksperimentus
plt.figure(figsize=(14, 5))
plt.bar(names, accuracies, color="steelblue")
plt.xticks(rotation=90)
plt.ylabel("Tikslumas (accuracy)")
plt.title("SVM eksperimentų tikslumas")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "accuracy_comparison.png")
plt.close()

# 2. Tikslumas pagal kernel tipą
kernel_types = ["rbf", "linear", "poly", "sigmoid"]
kernel_means = []
for k in kernel_types:
    vals = [accuracies[i] for i, e in enumerate(experiments)
            if e.hyperparameters.get("kernel") == k]
    kernel_means.append(sum(vals) / len(vals) if vals else 0)

plt.figure(figsize=(7, 5))
plt.bar(kernel_types, kernel_means, color=["steelblue", "coral", "green", "purple"])
plt.ylabel("Vidutinis tikslumas")
plt.title("Vidutinis tikslumas pagal kernel tipą")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "accuracy_by_kernel.png")
plt.close()

# 3. Tikslumas pagal C parametrą
plt.figure(figsize=(8, 5))
plt.scatter(c_values, accuracies, color="steelblue", alpha=0.7)
plt.xlabel("C parametras")
plt.ylabel("Tikslumas")
plt.title("Tikslumas pagal C parametrą")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "accuracy_by_C.png")
plt.close()

# 4. Tikslumo pasiskirstymas (histogram)
plt.figure(figsize=(7, 5))
plt.hist(accuracies, bins=8, color="steelblue", edgecolor="white")
plt.xlabel("Tikslumas")
plt.ylabel("Eksperimentų skaičius")
plt.title("Tikslumo pasiskirstymas")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "accuracy_distribution.png")
plt.close()

# 5. Standartinis nuokrypis per eksperimentus
plt.figure(figsize=(14, 5))
plt.bar(names, stds, color="coral")
plt.xticks(rotation=90)
plt.ylabel("Standartinis nuokrypis")
plt.title("Rezultatų stabilumas per eksperimentus")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "std_comparison.png")
plt.close()

print("Grafikai išsaugoti ui/static/charts/")