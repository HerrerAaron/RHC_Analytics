"""Phase 2: EDA, the naive RHC-vs-no-RHC mortality comparison, and
pre-adjustment covariate balance diagnostics."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("rhc_cleaned.csv")

# One-hot columns round-trip through CSV as True/False, not 0/1.
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# --- EDA: key variable summaries ------------------------------------------

key_numeric = ["age", "aps", "scoma1", "surv2md1", "das2d3pc"]
summary_stats = df[key_numeric].describe().T
summary_stats.to_csv("eda_summary_stats.csv")

treatment_counts = df["rhc_Yes"].value_counts().rename({0: "No RHC", 1: "RHC"})

category_cols = [c for c in df.columns if c.startswith("category_")]
category_counts = df[category_cols].sum()
category_counts["category_ARenalF"] = len(df) - category_counts.sum()
category_counts = category_counts.sort_values(ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

df["age"].plot(kind="hist", bins=30, ax=axes[0, 0])
axes[0, 0].set_title("Age distribution")
axes[0, 0].set_xlabel("Age (years)")

df["aps"].plot(kind="hist", bins=30, ax=axes[0, 1])
axes[0, 1].set_title("APACHE severity score (aps) distribution")
axes[0, 1].set_xlabel("aps")

category_counts.plot(kind="bar", ax=axes[1, 0])
axes[1, 0].set_title("Admission disease category")
axes[1, 0].tick_params(axis="x", rotation=75)

treatment_counts.plot(kind="bar", ax=axes[1, 1], color=["steelblue", "firebrick"])
axes[1, 1].set_title("Treatment group (RHC vs no RHC)")
axes[1, 1].tick_params(axis="x", rotation=0)

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/eda_key_variables.png", dpi=150)
plt.close(fig)

# --- Naive comparison -------------------------------------------------------

naive_mortality = df.groupby("rhc_Yes")["death_d30"].mean()
mortality_rhc = naive_mortality[1]
mortality_no_rhc = naive_mortality[0]

print(
    f"Naive 30-day mortality: RHC = {mortality_rhc:.1%}, "
    f"no RHC = {mortality_no_rhc:.1%}"
)

fig, ax = plt.subplots(figsize=(4, 5))
ax.bar(["No RHC", "RHC"], [mortality_no_rhc, mortality_rhc], color=["steelblue", "firebrick"])
ax.set_ylabel("30-day mortality rate")
ax.set_title("Naive (unadjusted) mortality comparison")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/naive_mortality_comparison.png", dpi=150)
plt.close(fig)

# --- Covariate balance -------------------------------------------------------


def standardized_mean_diff(treated, control):
    mean_diff = treated.mean() - control.mean()
    pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
    return mean_diff / pooled_sd if pooled_sd > 0 else np.nan


# Demographics, prognostic scores, comorbidity history, day-1 organ-system
# dysfunction flags, and day-1 vitals/labs: the same covariate classes used
# in the original Connors et al. propensity model.
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

treated_mask = df["rhc_Yes"] == 1
balance_rows = []
for col in covariates:
    treated_vals = df.loc[treated_mask, col]
    control_vals = df.loc[~treated_mask, col]
    balance_rows.append({
        "covariate": col,
        "mean_rhc": treated_vals.mean(),
        "mean_no_rhc": control_vals.mean(),
        "smd": standardized_mean_diff(treated_vals, control_vals),
    })

balance_table = pd.DataFrame(balance_rows)
balance_table["imbalanced"] = balance_table["smd"].abs() > 0.1
balance_table = balance_table.sort_values("smd", key=abs, ascending=False)
balance_table.to_csv("covariate_balance_naive.csv", index=False)

n_imbalanced = balance_table["imbalanced"].sum()
print(f"{n_imbalanced} of {len(balance_table)} covariates exceed |SMD| > 0.1 before adjustment")

fig, ax = plt.subplots(figsize=(7, 14))
plot_data = balance_table.sort_values("smd")
colors = ["firebrick" if v else "steelblue" for v in plot_data["imbalanced"]]
# Escape "$" so matplotlib doesn't parse income labels (e.g. "$25-$50k") as
# mathtext, which silently drops the dollar signs and remaps "-" to "−".
labels = plot_data["covariate"].str.replace("$", r"\$", regex=False)
ax.scatter(plot_data["smd"], labels, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.axvline(0.1, color="grey", linestyle="--", linewidth=0.8)
ax.axvline(-0.1, color="grey", linestyle="--", linewidth=0.8)
ax.set_xlabel("Standardized mean difference (RHC minus no RHC)")
ax.set_title("Covariate balance before adjustment")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/covariate_balance_love_plot.png", dpi=150)
plt.close(fig)
