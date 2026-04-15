import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GroupShuffleSplit

from src.features.build_features import extract_all_features


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent.parent / "data" / "raw" / "better_url_dataset.csv"


# -----------------------------
# 0) Load CSV -> urls, y
# -----------------------------
df = pd.read_csv(CSV_PATH)

df["url"] = df["url"].astype(str).str.strip()
df["type"] = df["type"].astype(str).str.lower().str.strip()

BAD_TOKENS = {"phishing", "malicious", "bad", "spam", "phish", "1", "true", "yes"}
GOOD_TOKENS = {"legitimate", "benign", "good", "clean", "0", "false", "no"}

mask = df["type"].isin(BAD_TOKENS | GOOD_TOKENS)
df = df[mask].copy()

urls = df["url"].tolist()
y = df["type"].isin(BAD_TOKENS).astype(int).to_numpy()

# TEMP LIMIT FOR TESTING
urls = urls[:5000]
y = y[:5000]

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
    rows = []
    for i, u in enumerate(urls_list):
        if i % 100 == 0:
            print(f"Processing URL {i}/{len(urls_list)}")
        rows.append(extract_all_features(u, use_page_features=True))

    Xdf = pd.DataFrame(rows)

    drop_cols = ["scheme", "netloc", "path", "params", "query", "fragment", "domain", "query_params"]
    Xdf.drop(columns=[c for c in drop_cols if c in Xdf.columns], inplace=True)

    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(0)
    Xdf = Xdf.select_dtypes(include=[np.number])
    return Xdf


X = build_X(urls)
y = np.array(y, dtype=int)


# -----------------------------
# 2) Domain-level Train/Valid/Test split
# -----------------------------
domains = []
for u in urls:
    parts = u.split("/")
    domains.append(parts[2] if len(parts) > 2 else u)

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(gss1.split(X, y, groups=domains))

X_train = X.iloc[train_idx]
y_train = y[train_idx]

X_temp = X.iloc[temp_idx]
y_temp = y[temp_idx]
domains_temp = [domains[i] for i in temp_idx]

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

MODEL_DIR = BASE_DIR.parent.parent / "models" / "trained"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "risklens_model.pkl"
COLS_PATH = MODEL_DIR / "feature_columns.pkl"

joblib.dump(clf, MODEL_PATH)
joblib.dump(list(X.columns), COLS_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Feature columns saved to: {COLS_PATH}")


# -----------------------------
# 5) Feature importances
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
# 6) Predictions
# -----------------------------
proba = clf.predict_proba(X_valid)[:, 1]

print("\nROC-AUC:", roc_auc_score(y_valid, proba))
print("PR-AUC :", average_precision_score(y_valid, proba))


# -----------------------------
# 7) Threshold helpers
# -----------------------------
def evaluate_threshold(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold_with_fpr_budget(y_true, probs, target_fpr=0.01):
    best_score = None
    best_stats = None

    for t in np.arange(0.10, 0.96, 0.01):
        stats = evaluate_threshold(y_true, probs, float(t))

        if stats["fpr"] <= target_fpr:
            score = (stats["recall"], stats["precision"], stats["f1"], -stats["threshold"])
            if best_score is None or score > best_score:
                best_score = score
                best_stats = stats

    if best_stats is None:
        for t in np.arange(0.10, 0.96, 0.01):
            stats = evaluate_threshold(y_true, probs, float(t))
            score = (stats["f1"], -stats["fpr"], stats["recall"])
            if best_score is None or score > best_score:
                best_score = score
                best_stats = stats

    return best_stats


# -----------------------------
# 8) Pick threshold
# -----------------------------
threshold_stats = choose_threshold_with_fpr_budget(y_valid, proba, target_fpr=0.01)
best_thr = threshold_stats["threshold"]

print("\nChosen threshold stats:")
print(threshold_stats)

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
# False-positive inspection
# -----------------------------
test_urls = [urls[i] for i in temp_idx]
test_urls = [test_urls[i] for i in test_idx]

results_df = pd.DataFrame({
    "url": test_urls,
    "y_true": y_test,
    "y_pred": pred_test,
    "y_prob": proba_test,
})

false_positives = results_df[
    (results_df["y_true"] == 0) & (results_df["y_pred"] == 1)
].copy()

print(f"\nFalse Positives: {len(false_positives)}")
if not false_positives.empty:
    print(false_positives.head(15).to_string(index=False))


# -----------------------------
# 9) Plots
# -----------------------------
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

pred_best = (proba >= best_thr).astype(int)
cm = confusion_matrix(y_valid, pred_best)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Phish"])
fig, ax = plt.subplots()
disp.plot(ax=ax, values_format="d")
ax.set_title(f"Confusion Matrix (thr={best_thr:.2f})")
plt.tight_layout()
plt.show()

topk = 20
gain_top = gain_imp.head(topk)[::-1]

plt.figure()
plt.barh(gain_top.index, gain_top.values)
plt.xlabel("Total Gain")
plt.title(f"Top {topk} Feature Importances (Gain)")
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(proba[y_valid == 0], bins=40, alpha=0.7, label="Legit")
plt.hist(proba[y_valid == 1], bins=40, alpha=0.7, label="Phish")
plt.xlabel("Predicted probability (phish)")
plt.ylabel("Count")
plt.title("Score Distributions (Validation)")
plt.legend()
plt.tight_layout()
plt.show()