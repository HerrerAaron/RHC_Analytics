"""Phase 4: predictive 30-day mortality risk model. Three scikit-learn
classifiers are hyperparameter-tuned and compared entirely within
cross-validation on the training set; only the single best-performing model
touches the held-out test set, exactly once, for final evaluation and
feature importance."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Paths are anchored to this script's own location, not the working
# directory, so it runs the same regardless of where it's invoked from.
ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_DIR / "rhc_cleaned.csv")

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

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

# max_iter is a solver convergence setting, not a tunable hyperparameter, so
# it's fixed rather than searched. C, n_estimators, max_depth, etc. do trade
# off bias vs. variance and are searched via GridSearchCV below. Grids are
# kept small deliberately: this is a hyperparameter search, not an exhaustive
# sweep.
model_specs = {
    "Logistic Regression": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        {"clf__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 5, 10],
        },
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
    ),
}

# --- Cross-validated hyperparameter search and model selection --------------

# Model comparison and hyperparameter tuning happen in the same step and use
# only X_train/y_train: for each model type, GridSearchCV searches its grid
# under 5-fold CV and keeps the best-scoring combination. The test set is not
# involved anywhere in this block.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

searches = {}
cv_rows = []
for name, (estimator, param_grid) in model_specs.items():
    search = GridSearchCV(estimator, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    searches[name] = search
    cv_rows.append({
        "model": name,
        "best_cv_roc_auc": search.best_score_,
        "best_params": search.best_params_,
    })

cv_results = pd.DataFrame(cv_rows)
cv_results.to_csv(CSV_DIR / "cv_model_comparison.csv", index=False)
print(cv_results.to_string(index=False))

# Single winner, selected purely on CV performance, before the test set is
# touched at all.
best_model_name = cv_results.loc[cv_results["best_cv_roc_auc"].idxmax(), "model"]
best_model = searches[best_model_name].best_estimator_
print(f"\nSelected model: {best_model_name}")

# --- Held-out test set evaluation (single model, single evaluation) ---------

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

test_metrics = pd.DataFrame([{
    "model": best_model_name,
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}])
test_metrics.to_csv(CSV_DIR / "test_set_metrics.csv", index=False)
print(test_metrics.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 6))
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
ax.plot(fpr, tpr, label=f"{best_model_name} (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title(f"ROC curve, {best_model_name} (held-out test set)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "roc_curves.png", dpi=150)
plt.close(fig)

# --- Feature importance (single model) ---------------------------------------

# Permutation importance also consumes the test set, so it's computed once,
# for the one selected model, not once per candidate model.
result = permutation_importance(
    best_model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1
)
importance_table = pd.DataFrame({
    "feature": covariates,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std,
}).sort_values("importance_mean", ascending=False)
importance_table.to_csv(CSV_DIR / "feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 10))
top = importance_table.head(20)
# Escape "$" so matplotlib doesn't parse income labels as mathtext.
labels = top["feature"].str.replace("$", r"\$", regex=False)
ax.barh(labels[::-1], top["importance_mean"][::-1])
ax.set_xlabel("Permutation importance (ROC-AUC drop)")
ax.set_title(f"Feature importance, {best_model_name}")
fig.tight_layout()
fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
plt.close(fig)
