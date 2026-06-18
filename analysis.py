"""
Diabetes Risk Prediction, end-to-end analysis
================================================
Predicts diabetes risk from health & lifestyle indicators (CDC BRFSS 2015),
then explains the model with SHAP.

Pipeline: load -> EDA -> preprocess -> train (baseline + model) ->
honest evaluation (ROC/PR/calibration) -> SHAP explainability -> save outputs.

Run:  python analysis.py
Outputs: figures + metrics.json in ./outputs
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless backend (save figures, no display)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
metrics = {}

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
df = pd.read_csv("data/diabetes_health_indicators.csv")
# Target is the binary diabetes flag; everything else is a predictor.
target = "Diabetes_binary" if "Diabetes_binary" in df.columns else df.columns[-1]
y = df[target].astype(int)
X = df.drop(columns=[target])
print(f"Rows: {len(df):,} | Features: {X.shape[1]} | Target: {target}")
print(f"Diabetes prevalence: {y.mean():.1%}")
metrics["n_rows"] = int(len(df))
metrics["n_features"] = int(X.shape[1])
metrics["prevalence"] = round(float(y.mean()), 4)

# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
# 2a. Class balance, important: the data is imbalanced, so accuracy alone lies.
plt.figure(figsize=(5, 4))
ax = sns.countplot(x=y, palette=["#9aa6b2", "#e31a1c"])
ax.set_xticklabels(["No diabetes", "Diabetes"])
ax.set_title("Class balance (target is imbalanced)"); ax.set_xlabel("")
for c in ax.containers:
    ax.bar_label(c, fmt="%d")
plt.tight_layout(); plt.savefig(OUT / "01_class_balance.png", dpi=120); plt.close()

# 2b. Key features vs. target (do diabetics differ on BMI / general health / age?)
key = [c for c in ["BMI", "GenHlth", "Age", "HighBP"] if c in X.columns]
fig, axes = plt.subplots(1, len(key), figsize=(4 * len(key), 4))
for ax, col in zip(np.atleast_1d(axes), key):
    sns.boxplot(x=y, y=X[col], ax=ax, palette=["#9aa6b2", "#e31a1c"])
    ax.set_xticklabels(["No", "Yes"]); ax.set_xlabel("Diabetes"); ax.set_title(col)
plt.tight_layout(); plt.savefig(OUT / "02_features_by_target.png", dpi=120); plt.close()

# 2c. Correlation of each feature with the target (which factors track diabetes?)
corr = df.corr(numeric_only=True)[target].drop(target).sort_values()
plt.figure(figsize=(7, 7))
corr.plot(kind="barh", color=np.where(corr > 0, "#e31a1c", "#1f78b4"))
plt.title(f"Correlation of each feature with {target}"); plt.tight_layout()
plt.savefig(OUT / "03_feature_correlation.png", dpi=120); plt.close()

# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (stratified to preserve class ratio)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

# ---------------------------------------------------------------------------
# 4. MODELS
#    Baseline: logistic regression (scaled, class-weighted for imbalance)
#    Main:     random forest (class-weighted; SHAP-friendly)
# ---------------------------------------------------------------------------
baseline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced"))

rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=20,
    class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)

models = {"Logistic Regression": baseline, "Random Forest": rf}
results = {}
for name, model in models.items():
    print(f"Training {name} ...")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "model": model,
        "proba": proba,
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }
    print(f"  ROC-AUC={results[name]['roc_auc']:.3f}  PR-AUC={results[name]['pr_auc']:.3f}")

metrics["models"] = {n: {"roc_auc": round(r["roc_auc"], 4),
                         "pr_auc": round(r["pr_auc"], 4)} for n, r in results.items()}

# Pick the best model by PR-AUC (more informative than ROC for imbalanced data)
best_name = max(results, key=lambda n: results[n]["pr_auc"])
best = results[best_name]
metrics["best_model"] = best_name
print(f"\nBest model: {best_name}")

# ---------------------------------------------------------------------------
# 5. HONEST EVALUATION
# ---------------------------------------------------------------------------
# 5a. ROC & Precision-Recall curves (both models)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["proba"])
    a1.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")
    prec, rec, _ = precision_recall_curve(y_test, r["proba"])
    a2.plot(rec, prec, label=f"{name} (AP={r['pr_auc']:.3f})")
a1.plot([0, 1], [0, 1], "k--", lw=1); a1.set(xlabel="False positive rate",
        ylabel="True positive rate", title="ROC curve"); a1.legend()
a2.axhline(y_test.mean(), ls="--", c="k", lw=1, label=f"baseline ({y_test.mean():.2f})")
a2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve"); a2.legend()
plt.tight_layout(); plt.savefig(OUT / "04_roc_pr_curves.png", dpi=120); plt.close()

# 5b. Choose a threshold that prioritises catching cases (recall) for screening.
#     We pick the threshold maximising F1, a balance of precision & recall.
prec, rec, thr = precision_recall_curve(y_test, best["proba"])
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_thr = float(thr[np.argmax(f1[:-1])])
metrics["chosen_threshold"] = round(best_thr, 3)
pred = (best["proba"] >= best_thr).astype(int)
print(f"\nChosen threshold (max F1): {best_thr:.3f}")
print(classification_report(y_test, pred, target_names=["No diabetes", "Diabetes"]))
metrics["classification_report"] = classification_report(
    y_test, pred, target_names=["No diabetes", "Diabetes"], output_dict=True)

# 5c. Confusion matrix at the chosen threshold
ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                       display_labels=["No diabetes", "Diabetes"]).plot(
    cmap="Reds", colorbar=False)
plt.title(f"{best_name}, confusion matrix (thr={best_thr:.2f})")
plt.tight_layout(); plt.savefig(OUT / "05_confusion_matrix.png", dpi=120); plt.close()

# 5d. Calibration, are predicted probabilities trustworthy?
frac_pos, mean_pred = calibration_curve(y_test, best["proba"], n_bins=10)
plt.figure(figsize=(6, 5))
plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
plt.plot(mean_pred, frac_pos, "o-", color="#e31a1c", label=best_name)
plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
plt.title("Calibration curve"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "06_calibration.png", dpi=120); plt.close()

# ---------------------------------------------------------------------------
# 6. EXPLAINABILITY (SHAP), why does the model predict risk?
# ---------------------------------------------------------------------------
import shap
print("\nComputing SHAP values (sampled) ...")
rf_model = results["Random Forest"]["model"]
X_shap = X_test.sample(min(2000, len(X_test)), random_state=RANDOM_STATE)
explainer = shap.TreeExplainer(rf_model)
sv = explainer.shap_values(X_shap)
# For binary RF, shap returns a list [class0, class1] (or 3D array); take class 1.
sv1 = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)

# 6a. Global importance (mean |SHAP|)
plt.figure()
shap.summary_plot(sv1, X_shap, plot_type="bar", show=False)
plt.title("Global feature importance (mean |SHAP|)")
plt.tight_layout(); plt.savefig(OUT / "07_shap_importance.png", dpi=120); plt.close()

# 6b. Beeswarm, direction & magnitude of each feature's effect
plt.figure()
shap.summary_plot(sv1, X_shap, show=False)
plt.title("How each feature pushes risk up or down")
plt.tight_layout(); plt.savefig(OUT / "08_shap_beeswarm.png", dpi=120); plt.close()

# ---------------------------------------------------------------------------
# 7. SAVE METRICS
# ---------------------------------------------------------------------------
with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nDone. Figures + metrics.json saved to ./{OUT}/")
