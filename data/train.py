import os, glob, warnings, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────
DATASET_DIR   = "natural_bank_datasets"
MODEL_DIR     = "models"
PRIMARY_MODEL = "rf"
ZIP_PATH      = "natural_bank_datasets.zip"

CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education",
                    "Self_Employed", "Property_Area", "Loan_Purpose"]
NUMERIC_COLS     = ["ApplicantIncome", "CoapplicantIncome",
                    "LoanAmount", "Loan_Amount_Term", "Credit_History"]
TARGET_COL   = "Loan_Status"
DROP_COLS    = ["Loan_ID"]

MODEL_LABELS = {
    "knn": "K-Nearest Neighbors",
    "nb":  "Naive Bayes",
    "dt":  "Decision Tree",
    "lr":  "Logistic Regression",
    "svm": "Support Vector Machine",
    "gb":  "Gradient Boosting",
    "rf":  "Random Forest",
}

os.makedirs(MODEL_DIR, exist_ok=True)

# ── EXTRACT DATASET ─────────────────────────────────────────
if os.path.exists(ZIP_PATH):
    print(f"Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(".")
    print(f"Done — extracted to '{DATASET_DIR}/'")
elif os.path.exists(DATASET_DIR):
    print("Dataset folder found, skipping extraction")
else:
    raise FileNotFoundError(f"Neither '{ZIP_PATH}' nor '{DATASET_DIR}/' found.")

csv_files  = sorted([
    f for f in glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    if "SUMMARY" not in os.path.basename(f).upper()
])
bank_names = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
print(f"{len(csv_files)} bank datasets found\n")

# ── PREPROCESSING ────────────────────────────────────────────
def preprocess_bank_data(df: pd.DataFrame):
    df = df.copy()
    for col in DROP_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    encoders = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    feature_names = list(df.drop(columns=[TARGET_COL]).columns)
    X = df.drop(columns=[TARGET_COL]).values
    y = df[TARGET_COL].values
    return X, y, encoders, feature_names


def fill_nan(X_tr, X_te):
    medians = np.nanmedian(X_tr, axis=0)
    for i in range(X_tr.shape[1]):
        X_tr[np.isnan(X_tr[:, i]), i] = medians[i]
        X_te[np.isnan(X_te[:, i]), i] = medians[i]
    return X_tr, X_te


def get_models():
    return {
        "knn": KNeighborsClassifier(n_neighbors=15),
        "nb":  GaussianNB(),
        "dt":  DecisionTreeClassifier(random_state=42),
        "lr":  LogisticRegression(max_iter=500, random_state=42),
        "svm": SVC(kernel="linear", probability=True, random_state=42),
        "gb":  GradientBoostingClassifier(random_state=42),
        "rf":  RandomForestClassifier(n_estimators=100, random_state=42),
    }

# ── TRAIN ALL BANKS ──────────────────────────────────────────
training_results = []
print(f"Training {MODEL_LABELS[PRIMARY_MODEL]} on {len(bank_names)} banks...\n")
print(f"{'Bank':<38} {'Samples':>8} {'Accuracy':>10} {'F1':>8}  Status")
print("-" * 75)

for csv_path, bank_name in zip(csv_files, bank_names):
    try:
        df = pd.read_csv(csv_path)
        X, y, encoders, feature_names = preprocess_bank_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_test = fill_nan(X_train, X_test)
        model = get_models()[PRIMARY_MODEL]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        bundle = {
            "model":         model,
            "encoders":      encoders,
            "feature_names": feature_names,
            "bank_name":     bank_name,
            "accuracy":      acc,
            "f1":            f1,
        }
        joblib.dump(bundle, os.path.join(MODEL_DIR, f"{bank_name}_model.pkl"))
        training_results.append({"bank": bank_name, "samples": len(df),
                                  "accuracy": acc, "f1": f1, "ok": True})
        print(f"{bank_name:<38} {len(df):>8} {acc:>10.3f} {f1:>8.3f}  OK")
    except Exception as e:
        training_results.append({"bank": bank_name, "samples": 0,
                                  "accuracy": 0, "f1": 0, "ok": False})
        print(f"{bank_name:<38} {'ERR':>8} {'':>10} {'':>8}  ERROR: {e}")

results_df = pd.DataFrame(training_results)
ok_count   = results_df["ok"].sum()
print("-" * 75)
print(f"\nDone — {ok_count}/{len(results_df)} banks trained successfully")
print(f"Models saved in '{MODEL_DIR}/'")
print(f"Average Accuracy : {results_df['accuracy'].mean():.3f}")
print(f"Average F1-Score : {results_df['f1'].mean():.3f}")

# ── TRAINING RESULTS CHART ───────────────────────────────────
ok = results_df[results_df["ok"]].copy().sort_values("accuracy", ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle("Multi-Bank Training Results — LoanSathi", fontsize=16, fontweight="bold")
colors = ["#2ecc71" if v >= 0.80 else "#f39c12" if v >= 0.70 else "#e74c3c"
          for v in ok["accuracy"]]
axes[0].barh(ok["bank"], ok["accuracy"], color=colors, edgecolor="white", height=0.7)
axes[0].axvline(ok["accuracy"].mean(), color="#2c3e50", linestyle="--", linewidth=1.5,
                label=f"Avg = {ok['accuracy'].mean():.3f}")
axes[0].set_xlabel("Accuracy")
axes[0].set_title("Per-Bank Model Accuracy", fontsize=13, fontweight="bold")
axes[0].set_xlim(0.5, 1.0)
axes[0].tick_params(axis="y", labelsize=7.5)
axes[0].legend(handles=[
    mpatches.Patch(color="#2ecc71", label=">= 80%"),
    mpatches.Patch(color="#f39c12", label="70-80%"),
    mpatches.Patch(color="#e74c3c", label="< 70%"),
], loc="lower right", fontsize=9)
axes[1].hist(ok["accuracy"], bins=15, color="#3498db", edgecolor="white", alpha=0.85)
axes[1].axvline(ok["accuracy"].mean(),   color="#e74c3c", linestyle="--", linewidth=2,
                label=f"Mean = {ok['accuracy'].mean():.3f}")
axes[1].axvline(ok["accuracy"].median(), color="#2ecc71", linestyle="--", linewidth=2,
                label=f"Median = {ok['accuracy'].median():.3f}")
axes[1].set_xlabel("Accuracy")
axes[1].set_ylabel("Number of Banks")
axes[1].set_title("Distribution of Accuracies", fontsize=13, fontweight="bold")
axes[1].legend()
plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()

# ── MODEL COMPARISON ON SBI ──────────────────────────────────
df_sbi = pd.read_csv(os.path.join(DATASET_DIR, "SBI.csv"))
X_s, y_s, enc_s, feat_s = preprocess_bank_data(df_sbi)
X_tr, X_te, y_tr, y_te  = train_test_split(X_s, y_s, test_size=0.2, random_state=42, stratify=y_s)
X_tr, X_te = fill_nan(X_tr, X_te)

model_perf = {}
all_models = get_models()
print(f"\n{'Model':<25} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
print("-" * 70)
for key, mdl in all_models.items():
    mdl.fit(X_tr, y_tr)
    yp  = mdl.predict(X_te)
    acc = accuracy_score(y_te, yp)
    f1  = f1_score(y_te, yp, average="weighted")
    pr  = precision_score(y_te, yp, average="weighted", zero_division=0)
    rc  = recall_score(y_te, yp, average="weighted", zero_division=0)
    model_perf[key] = {"name": MODEL_LABELS[key], "acc": acc, "f1": f1, "pr": pr, "rc": rc}
    print(f"{MODEL_LABELS[key]:<25} {acc:>10.3f} {f1:>10.3f} {pr:>10.3f} {rc:>10.3f}")

best_key = max(model_perf, key=lambda k: model_perf[k]["acc"])
print(f"\nBest: {MODEL_LABELS[best_key]} ({model_perf[best_key]['acc']*100:.1f}% accuracy)")

# confusion matrices
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("Confusion Matrices — All 7 Models on SBI", fontsize=15, fontweight="bold")
axes_flat = axes.flatten()
for i, (key, mdl) in enumerate(all_models.items()):
    yp = mdl.predict(X_te)
    cm = confusion_matrix(y_te, yp)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes_flat[i],
                cbar=False, linewidths=0.5,
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"])
    axes_flat[i].set_title(
        f"{MODEL_LABELS[key]}\nAcc={model_perf[key]['acc']:.3f}  F1={model_perf[key]['f1']:.3f}",
        fontsize=10, fontweight="bold")
    axes_flat[i].set_xlabel("Predicted")
    axes_flat[i].set_ylabel("Actual")
axes_flat[-1].set_visible(False)
plt.tight_layout()
plt.show()

# accuracy & F1 comparison
fig, ax = plt.subplots(figsize=(13, 5))
model_names = [model_perf[k]["name"] for k in all_models]
accs = [model_perf[k]["acc"] for k in all_models]
f1s  = [model_perf[k]["f1"]  for k in all_models]
x    = np.arange(len(model_names))
w    = 0.35
bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#3498db", edgecolor="white")
bars2 = ax.bar(x + w/2, f1s,  w, label="F1-Score",  color="#2ecc71", edgecolor="white")
for b in list(bars1) + list(bars2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
            f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha="right")
ax.set_ylim(0.5, 1.05)
ax.set_title("All 7 Models — Accuracy & F1-Score on SBI", fontsize=13, fontweight="bold")
ax.legend()
ax.set_ylabel("Score")
plt.tight_layout()
plt.show()

# correlation heatmap
fig, ax = plt.subplots(figsize=(13, 9))
df_enc = df_sbi.drop(columns=["Loan_ID"], errors="ignore").copy()
for col in df_enc.select_dtypes(include=["object", "string"]).columns:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
df_enc.fillna(df_enc.median(numeric_only=True), inplace=True)
sns.heatmap(df_enc.corr(), annot=True, fmt=".2f", cmap="BrBG",
            linewidths=1, ax=ax, annot_kws={"size": 9})
ax.set_title("Feature Correlation Heatmap — SBI Dataset", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\nAll done. Place the '{MODEL_DIR}/' folder inside mysite/ and run the app.")
