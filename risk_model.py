"""Phase 4: predictive 30-day mortality risk model, compared across three
scikit-learn classifiers, evaluated on a held-out test set, and interpreted
via permutation feature importance."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("rhc_cleaned.csv")

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Same covariate set validated for balance/propensity in Phases 2-3, minus
# rhc_Yes: this is a pure admission-characteristics risk score, not a
# treatment-effect model, so the treatment itself isn't a predictor.
category_cols = [c for c in df.columns if c.startswith("category_")]
covariates = (
    ["age", "edu", "sex_Male", "race_Black", "race_Other"]
    + ["income_$11-$25k", "income_$25-$50k", "income_> $50k"]
    + [c for c in df.columns if c.startswith("insurance_class_")]
    + ["surv2md1", "das2d3pc", "aps", "scoma1"]
    + ["cardiohx", "chfhx", "dementhx", "psychhx", "chrpulhx", "renalhx",
       "liverhx", "gibledhx", "malighx", "immunhx", "transhx", "amihx"]
    + ["resp", "card", "neuro", "gastr", "renal", "meta", "hema", "seps",
       "trauma", "ortho"]
    + ["meanbp1", "wblc1", "hrt1", "resp1", "temp1", "pafi1", "alb1",
       "hema1", "bili1", "crea1", "sod1", "pot1", "paco21", "ph1"]
    + ["weight_kg", "no_resus_Yes"]
    + category_cols
)

X = df[covariates]
y = df["death_d30"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Logistic regression needs standardized inputs to converge reliably and for
# coefficient magnitudes to be comparable; tree ensembles are scale-invariant
# and split on raw values directly, so only LR is wrapped in a scaler.
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# --- Cross-validated model comparison ----------------------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_rows.append({
        "model": name,
        "cv_roc_auc_mean": scores.mean(),
        "cv_roc_auc_std": scores.std(),
    })
cv_results = pd.DataFrame(cv_rows)
cv_results.to_csv("cv_model_comparison.csv", index=False)
print(cv_results.to_string(index=False))

# --- Held-out test set evaluation --------------------------------------------

fitted_models = {}
test_rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    test_rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    })
test_results = pd.DataFrame(test_rows)
test_results.to_csv("test_set_metrics.csv", index=False)
print(test_results.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 6))
for name, model in fitted_models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves (held-out test set)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/roc_curves.png", dpi=150)
plt.close(fig)

# --- Feature importance -------------------------------------------------------

# Permutation importance (not each model's built-in importance): impurity-
# based importance is biased toward high-cardinality/continuous features, so
# using one model-agnostic method keeps the three models comparable.
importance_rows = []
for name, model in fitted_models.items():
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1
    )
    for feature, mean_imp, std_imp in zip(covariates, result.importances_mean, result.importances_std):
        importance_rows.append({
            "model": name,
            "feature": feature,
            "importance_mean": mean_imp,
            "importance_std": std_imp,
        })
importance_table = pd.DataFrame(importance_rows)
importance_table.to_csv("feature_importance.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for ax, name in zip(axes, fitted_models):
    top = (
        importance_table[importance_table["model"] == name]
        .sort_values("importance_mean", ascending=False)
        .head(15)
    )
    # Escape "$" so matplotlib doesn't parse income labels as mathtext.
    labels = top["feature"].str.replace("$", r"\$", regex=False)
    ax.barh(labels[::-1], top["importance_mean"][::-1])
    ax.set_title(name)
    ax.set_xlabel("Permutation importance (ROC-AUC drop)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/feature_importance.png", dpi=150)
plt.close(fig)
