"""Phase 3: propensity score model, IPW-adjusted 30-day mortality effect,
and post-weighting covariate balance check."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Paths are anchored to this script's own location, not the working
# directory, so it runs the same regardless of where it's invoked from.
ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_DIR / "rhc_cleaned.csv")

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

# Same covariate set used for the Phase 2 balance check: demographics,
# prognostic severity scores, comorbidity history, day-1 organ-system
# flags, day-1 vitals/labs, and disease category.
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

# --- Propensity score model --------------------------------------------------

X = sm.add_constant(df[covariates])
y = df["rhc_Yes"]

ps_model = sm.Logit(y, X).fit(disp=False)
df["propensity_score"] = ps_model.predict(X)

ps_model.summary2().tables[1].to_csv(CSV_DIR / "propensity_model_coefficients.csv")

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(df.loc[df["rhc_Yes"] == 1, "propensity_score"], bins=30, alpha=0.6, label="RHC")
ax.hist(df.loc[df["rhc_Yes"] == 0, "propensity_score"], bins=30, alpha=0.6, label="No RHC")
ax.set_xlabel("Estimated propensity score P(RHC | covariates)")
ax.set_ylabel("Number of patients")
ax.set_title("Propensity score overlap by treatment group")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "propensity_score_overlap.png", dpi=150)
plt.close(fig)

# --- Inverse probability weights ---------------------------------------------

# Stabilized weights (numerator = marginal treatment probability) instead of
# raw 1/PS, to curb variance from patients with propensity scores near 0/1
# without changing what the weights estimate.
p_treated = df["rhc_Yes"].mean()
df["ipw"] = np.where(
    df["rhc_Yes"] == 1,
    p_treated / df["propensity_score"],
    (1 - p_treated) / (1 - df["propensity_score"]),
)

print(f"IPW weight range: {df['ipw'].min():.2f} to {df['ipw'].max():.2f}")

# --- Post-weighting covariate balance ----------------------------------------


def weighted_smd(values, treated_mask, weights):
    t_vals, t_w = values[treated_mask], weights[treated_mask]
    c_vals, c_w = values[~treated_mask], weights[~treated_mask]
    t_mean = np.average(t_vals, weights=t_w)
    c_mean = np.average(c_vals, weights=c_w)
    t_var = np.average((t_vals - t_mean) ** 2, weights=t_w)
    c_var = np.average((c_vals - c_mean) ** 2, weights=c_w)
    pooled_sd = np.sqrt((t_var + c_var) / 2)
    return (t_mean - c_mean) / pooled_sd if pooled_sd > 0 else np.nan


treated_mask = df["rhc_Yes"] == 1
weighted_rows = [
    {"covariate": col, "smd_weighted": weighted_smd(df[col].to_numpy(), treated_mask.to_numpy(), df["ipw"].to_numpy())}
    for col in covariates
]
weighted_balance = pd.DataFrame(weighted_rows)

naive_balance = pd.read_csv(CSV_DIR / "covariate_balance_naive.csv").rename(columns={"smd": "smd_unweighted"})
balance_table = naive_balance[["covariate", "smd_unweighted"]].merge(weighted_balance, on="covariate")
balance_table["imbalanced_before"] = balance_table["smd_unweighted"].abs() > 0.1
balance_table["imbalanced_after"] = balance_table["smd_weighted"].abs() > 0.1
balance_table = balance_table.sort_values("smd_weighted", key=abs, ascending=False)
balance_table.to_csv(CSV_DIR / "covariate_balance_weighted.csv", index=False)

n_before = balance_table["imbalanced_before"].sum()
n_after = balance_table["imbalanced_after"].sum()
print(f"Imbalanced covariates (|SMD| > 0.1): {n_before} before weighting, {n_after} after weighting")

fig, ax = plt.subplots(figsize=(7, 14))
plot_data = balance_table.sort_values("smd_unweighted")
labels = plot_data["covariate"].str.replace("$", r"\$", regex=False)
ax.scatter(plot_data["smd_unweighted"], labels, color="firebrick", label="Before weighting", alpha=0.7)
ax.scatter(plot_data["smd_weighted"], labels, color="steelblue", label="After weighting", alpha=0.7)
ax.axvline(0, color="black", linewidth=0.8)
ax.axvline(0.1, color="grey", linestyle="--", linewidth=0.8)
ax.axvline(-0.1, color="grey", linestyle="--", linewidth=0.8)
ax.set_xlabel("Standardized mean difference (RHC minus no RHC)")
ax.set_title("Covariate balance before vs. after IPW")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "covariate_balance_before_after.png", dpi=150)
plt.close(fig)

# --- IPW-adjusted treatment effect -------------------------------------------

# IPW already balances the covariates, so the outcome regression only needs
# the treatment indicator; its coefficient is the IPW estimator of the ATE.
# HC1 (robust) SEs are required here: the weights themselves induce
# heteroskedasticity that OLS-style standard errors would understate.
X_outcome = sm.add_constant(df["rhc_Yes"])
outcome_model = sm.WLS(df["death_d30"], X_outcome, weights=df["ipw"]).fit(cov_type="HC1")

ate = outcome_model.params["rhc_Yes"]  # ATE: average treatment effect
ci_low, ci_high = outcome_model.conf_int().loc["rhc_Yes"]

naive_ate = df.loc[df["rhc_Yes"] == 1, "death_d30"].mean() - df.loc[df["rhc_Yes"] == 0, "death_d30"].mean()

print(f"Naive risk difference: {naive_ate:+.1%}")
print(f"IPW-adjusted risk difference: {ate:+.1%} (95% CI: {ci_low:+.1%} to {ci_high:+.1%})")
