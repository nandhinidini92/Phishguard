"""End-to-end pipeline: train -> evaluate -> SHAP -> drift detection -> adaptive retraining.

Usage:
  python run_experiments.py --demo                      # synthetic smoke test (NOT for publication)
  python run_experiments.py --train data/train.csv --drift data/later.csv   # real data
"""
import argparse, json, time, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve)
from src.data import load_csv, make_synthetic
from src.models import get_models
from src.explain import shap_summary
from src.drift import drift_report

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true")
ap.add_argument("--train"); ap.add_argument("--drift")
ap.add_argument("--out", default="outputs")
a = ap.parse_args()

if a.demo:  # watermark every figure so synthetic output can never pass as real results
    _orig = plt.savefig
    def _wm(*args, **kw):
        plt.gcf().text(.5, .5, "SYNTHETIC DEMO - NOT REAL RESULTS", ha="center", va="center",
                       rotation=25, fontsize=14, color="red", alpha=.25)
        return _orig(*args, **kw)
    plt.savefig = _wm

if a.demo:
    X, y, _ = make_synthetic(8000, seed=1)
    Xd, yd, _ = make_synthetic(3000, drift=0.45, seed=2)
else:
    X, y = load_csv(a.train); Xd, yd = load_csv(a.drift)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---- 1. Baseline comparison -------------------------------------------------
res, probas, fitted = [], {}, {}
for name, m in get_models().items():
    t0 = time.time(); m.fit(Xtr, ytr); tt = time.time() - t0
    p = m.predict_proba(Xte)[:, 1]; pred = (p >= .5).astype(int)
    cv = cross_val_score(m, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1")
    res.append(dict(model=name, accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
                    recall=recall_score(yte, pred), f1=f1_score(yte, pred), auc=roc_auc_score(yte, p),
                    cv_f1_mean=cv.mean(), cv_f1_std=cv.std(), train_sec=tt))
    probas[name], fitted[name] = p, m
R = pd.DataFrame(res).round(4); R.to_csv(f"{a.out}/table_model_comparison.csv", index=False)
print("\n=== Model comparison ===\n", R.to_string(index=False))

# ROC + confusion matrix (screenshots for the paper)
plt.figure(figsize=(5, 4))
for n, p in probas.items():
    fpr, tpr, _ = roc_curve(yte, p); plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(yte,p):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=.7); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.legend(fontsize=7); plt.title("ROC curves"); plt.tight_layout(); plt.savefig(f"{a.out}/fig_roc.png", dpi=160); plt.close()

best = R.sort_values("f1", ascending=False).iloc[0]["model"]
ConfusionMatrixDisplay(confusion_matrix(yte, (probas[best] >= .5).astype(int)),
                       display_labels=["Legit", "Phish"]).plot(cmap="Blues")
plt.title(f"Confusion matrix – {best}"); plt.tight_layout()
plt.savefig(f"{a.out}/fig_confusion.png", dpi=160); plt.close()

# ---- 2. Explainability -------------------------------------------------------
model = fitted["XGBoost"]
imp = shap_summary(model, Xtr, Xte.sample(min(500, len(Xte)), random_state=0), f"{a.out}/fig_shap_summary.png")
pd.Series(imp, index=X.columns).sort_values(ascending=False).round(4).to_csv(f"{a.out}/table_shap_importance.csv")

# ---- 3. Drift detection ------------------------------------------------------
rep, alarm = drift_report(Xtr, Xd)
rep.round(4).to_csv(f"{a.out}/table_drift_report.csv", index=False)
print(f"\n=== Drift alarm: {alarm} ===\n", rep.head(6).round(3).to_string(index=False))
plt.figure(figsize=(6, 4)); top = rep.head(10)[::-1]
plt.barh(top["feature"], top["psi"]); plt.axvline(.2, color="r", ls="--", label="PSI=0.2")
plt.xlabel("PSI"); plt.legend(); plt.title("Feature drift (top 10)"); plt.tight_layout()
plt.savefig(f"{a.out}/fig_drift_psi.png", dpi=160); plt.close()

# ---- 4. Adaptive retraining --------------------------------------------------
Xd_a, Xd_b, yd_a, yd_b = train_test_split(Xd, yd, test_size=.5, stratify=yd, random_state=7)
def score(m, Xs, ys):
    p = m.predict_proba(Xs)[:, 1]; pr = (p >= .5).astype(int)
    return dict(f1=f1_score(ys, pr), recall=recall_score(ys, pr), auc=roc_auc_score(ys, p))
before = score(model, Xd_b, yd_b)
from xgboost import XGBClassifier
adapt = XGBClassifier(n_estimators=250, max_depth=6, learning_rate=.1, eval_metric="logloss", random_state=42)
# Strategy: retrain on old data + drifted window, up-weighting recent samples
w = np.r_[np.ones(len(Xtr)), np.full(len(Xd_a), 3.0)]
adapt.fit(pd.concat([Xtr, Xd_a]), np.r_[ytr, yd_a], sample_weight=w)
after = score(adapt, Xd_b, yd_b)
A = pd.DataFrame([dict(stage="Static model (no adaptation)", **before),
                  dict(stage="Drift-triggered retraining", **after)]).round(4)
A.to_csv(f"{a.out}/table_adaptation.csv", index=False)
print("\n=== Adaptation ===\n", A.to_string(index=False))
plt.figure(figsize=(4.5, 3.5)); x = np.arange(3); wd = .35
plt.bar(x - wd/2, list(before.values()), wd, label="Static"); plt.bar(x + wd/2, list(after.values()), wd, label="Adapted")
plt.xticks(x, ["F1", "Recall", "AUC"]); plt.ylim(0, 1.05); plt.legend(); plt.title("Performance on drifted data")
plt.tight_layout(); plt.savefig(f"{a.out}/fig_adaptation.png", dpi=160); plt.close()
json.dump(dict(best_model=best, drift_alarm=alarm), open(f"{a.out}/summary.json", "w"))
print("\nSaved tables/figures to", a.out)
