import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.features.url_features import extract_all_features
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
import matplotlib.pyplot as plt




# -----------------------------
# 0) Load CSV -> urls, y
# -----------------------------
import os

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw", "better_url_dataset.csv")
df = pd.read_csv(CSV_PATH)

df["url"] = df["url"].astype(str).str.strip()
df["type"] = df["type"].astype(str).str.lower().str.strip()

BAD_TOKENS  = {"phishing", "malicious", "bad", "spam", "phish", "1", "true", "yes"}
GOOD_TOKENS = {"legitimate", "benign", "good", "clean", "0", "false", "no"}

mask = df["type"].isin(BAD_TOKENS | GOOD_TOKENS)
df = df[mask].copy()

urls = df["url"].tolist()
y = df["type"].isin(BAD_TOKENS).astype(int).to_numpy()
urls = urls[:5000] # Temp data limt
y = y[:5000] # temp data limit 
total = len(y)
counts = np.bincount(y)

print("\n===== FULL DATASET DISTRIBUTION =====")
print(f"Legitimate (0): {counts[0]} ({counts[0]/total:.2%})")
print(f"Phishing   (1): {counts[1]} ({counts[1]/total:.2%})")
print("Loaded rows:", len(df))
print("Label counts:", np.bincount(y))


# -----------------------------
# 1) Build feature matrix
# -----------------------------
def build_X(urls_list):
    rows = [extract_all_features(u) for u in urls_list]
    Xdf = pd.DataFrame(rows)

    drop_cols = ["scheme", "netloc", "path", "params", "query", "fragment", "domain", "query_params"]
    Xdf.drop(columns=[c for c in drop_cols if c in Xdf.columns], inplace=True)

    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(0)
    Xdf = Xdf.select_dtypes(include=[np.number])
    return Xdf

X = build_X(urls)
y = np.array(y, dtype=int)


# -----------------------------
# 2) Domain-level train/valid split
# -----------------------------
# -----------------------------
# 2) Domain-level Train/Valid/Test split
# -----------------------------
domains = [u.split("/")[2] for u in urls]

# First split: Train (70%) + Temp (30%)
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(gss1.split(X, y, groups=domains))

X_train = X.iloc[train_idx]
y_train = y[train_idx]

X_temp = X.iloc[temp_idx]
y_temp = y[temp_idx]
domains_temp = [domains[i] for i in temp_idx]

# Second split: Validation (15%) + Test (15%)
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
valid_idx, test_idx = next(gss2.split(X_temp, y_temp, groups=domains_temp))

X_valid = X_temp.iloc[valid_idx]
y_valid = y_temp[valid_idx]

X_test = X_temp.iloc[test_idx]
y_test = y_temp[test_idx]
def show_distribution(name, labels):
    total = len(labels)
    counts = np.bincount(labels)
    print(f"\n{name} distribution:")
    print(f"  Legit (0): {counts[0]} ({counts[0]/total:.2%})")
    print(f"  Phish (1): {counts[1]} ({counts[1]/total:.2%})")

show_distribution("TRAIN", y_train)
show_distribution("VALID", y_valid)
show_distribution("TEST", y_test)
print("Train size:", len(X_train))
print("Valid size:", len(X_valid))
print("Test size :", len(X_test))


# -----------------------------
# 3) Handle class imbalance
# -----------------------------
neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
scale_pos_weight = neg / max(1, pos)


# -----------------------------
# 4) Train LightGBM
# -----------------------------
clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    force_row_wise=True,
    verbosity=-1
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(stopping_rounds=150),
        lgb.log_evaluation(period=50)
    ]
)

# -----------------------------
# Save trained model (.pkl)
# -----------------------------
import joblib

MODEL_DIR = BASE_DIR.parent.parent / "models" / "trained"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "risklens_model.pkl"
COLS_PATH = MODEL_DIR / "feature_columns.pkl"

joblib.dump(clf, MODEL_PATH)
joblib.dump(list(X.columns), COLS_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Feature columns saved to: {COLS_PATH}")
# -----------------------------
# 5) Feature importances (AFTER fit)
# -----------------------------
booster = clf.booster_

split_imp = pd.Series(
    booster.feature_importance(importance_type="split"),
    index=X.columns
).sort_values(ascending=False)

gain_imp = pd.Series(
    booster.feature_importance(importance_type="gain"),
    index=X.columns
).sort_values(ascending=False)

print("\nTop 25 importances (split count):\n", split_imp.head(25))
print("\nTop 25 importances (gain):\n", gain_imp.head(25))


# -----------------------------
# 6) Predictions (uncalibrated)
# -----------------------------
proba = clf.predict_proba(X_valid)[:, 1]

print("\nROC-AUC:", roc_auc_score(y_valid, proba))
print("PR-AUC :", average_precision_score(y_valid, proba))


# -----------------------------
# 7) Optional: calibrated probabilities
#     (calibrate AFTER the model is trained)
# -----------------------------
USE_CALIBRATION = False

if USE_CALIBRATION:
    from sklearn.calibration import CalibratedClassifierCV
    cal = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    cal.fit(X_train, y_train)
    proba = cal.predict_proba(X_valid)[:, 1]
    print("\n[Calibrated]")
    print("ROC-AUC:", roc_auc_score(y_valid, proba))
    print("PR-AUC :", average_precision_score(y_valid, proba))


# -----------------------------
# 8) Pick best threshold (max F1)
# -----------------------------
thresholds = np.linspace(0.05, 0.95, 19)
best_thr, best_f1 = 0.5, -1.0

for t in thresholds:
    p = (proba >= t).astype(int)
    tp = int(((p == 1) & (y_valid == 1)).sum())
    fp = int(((p == 1) & (y_valid == 0)).sum())
    fn = int(((p == 0) & (y_valid == 1)).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    if f1 > best_f1:
        best_thr, best_f1 = float(t), float(f1)

print("\nBest F1 threshold:", best_thr, "F1:", best_f1)

pred = (proba >= best_thr).astype(int)
print("Confusion matrix:\n", confusion_matrix(y_valid, pred))
print(classification_report(y_valid, pred, digits=4, zero_division=0))

# -----------------------------
# FINAL TEST EVALUATION
# -----------------------------
print("\n===== FINAL TEST SET RESULTS =====")

proba_test = clf.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= best_thr).astype(int)

print("ROC-AUC:", roc_auc_score(y_test, proba_test))
print("PR-AUC :", average_precision_score(y_test, proba_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_test))
print(classification_report(y_test, pred_test, digits=4, zero_division=0))

# -----------------------------
# 9) Plots for presentation
# -----------------------------
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay

# Use the proba you already computed (either calibrated or not)
# proba = clf.predict_proba(X_valid)[:, 1]  # already done earlier

# ---- (A) ROC Curve ----
fpr, tpr, _ = roc_curve(y_valid, proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ---- (B) Confusion Matrix at best threshold ----
pred_best = (proba >= best_thr).astype(int)
cm = confusion_matrix(y_valid, pred_best)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Phish"])
fig, ax = plt.subplots()
disp.plot(ax=ax, values_format="d")
ax.set_title(f"Confusion Matrix (thr={best_thr:.2f})")
plt.tight_layout()
plt.show()

# ---- (C) Feature Importance (GAIN) Top 20 ----
topk = 20
gain_top = gain_imp.head(topk)[::-1]  # reverse for nice horizontal bars

plt.figure()
plt.barh(gain_top.index, gain_top.values)
plt.xlabel("Total Gain")
plt.title(f"Top {topk} Feature Importances (Gain)")
plt.tight_layout()
plt.show()

# ---- (D) Optional: Probability distribution by class ----
plt.figure()
plt.hist(proba[y_valid == 0], bins=40, alpha=0.7, label="Legit")
plt.hist(proba[y_valid == 1], bins=40, alpha=0.7, label="Phish")
plt.xlabel("Predicted probability (phish)")
plt.ylabel("Count")
plt.title("Score Distributions (Validation)")
plt.legend()
plt.tight_layout()
plt.show()