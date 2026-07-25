"""Phase 6: Kaplan-Meier survival curves by treatment group, and a Cox
proportional hazards model of 30-day time-to-death."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test

# Paths are anchored to this script's own location, not the working
# directory, so it runs the same regardless of where it's invoked from.
ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_DIR / "rhc_cleaned.csv")

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

# --- Kaplan-Meier survival curves --------------------------------------

# survial_d30 (days to death or censoring) and death_d30 (1 = death
# observed, 0 = censored at day 30) were derived from the raw admission and
# death dates in Phase 1, not taken as given from the dataset.
treated = df[df["rhc_Yes"] == 1]
control = df[df["rhc_Yes"] == 0]

km_treated = KaplanMeierFitter(label="RHC")
km_treated.fit(treated["survial_d30"], event_observed=treated["death_d30"])

km_control = KaplanMeierFitter(label="No RHC")
km_control.fit(control["survial_d30"], event_observed=control["death_d30"])

fig, ax = plt.subplots(figsize=(7, 5))
km_treated.plot_survival_function(ax=ax)
km_control.plot_survival_function(ax=ax)
ax.set_xlabel("Days since admission")
ax.set_ylabel("Proportion surviving")
ax.set_title("30-day survival by treatment group")
fig.tight_layout()
fig.savefig(FIG_DIR / "kaplan_meier_survival.png", dpi=150)
plt.close(fig)

# Log-rank test: formal test of whether the two survival curves differ
# overall, the time-to-event equivalent of the naive comparison in Phase 2,
# still unadjusted for confounding.
logrank_result = logrank_test(
    treated["survial_d30"], control["survial_d30"],
    event_observed_A=treated["death_d30"], event_observed_B=control["death_d30"],
)
print(f"Log-rank test: statistic={logrank_result.test_statistic:.2f}, p={logrank_result.p_value:.4f}")

# --- Cox proportional hazards model -------------------------------------

# Same covariate set used throughout Phases 2-5, plus rhc_Yes itself: unlike
# the Phase 4 risk model (which deliberately excludes treatment to keep it a
# pure admission risk score), the point here is specifically to read off
# RHC's own hazard ratio alongside every other risk factor.
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
cox_covariates = ["rhc_Yes"] + covariates

cox_data = df[cox_covariates + ["survial_d30", "death_d30"]]
cox_model = CoxPHFitter()
cox_model.fit(cox_data, duration_col="survial_d30", event_col="death_d30")

cox_model.summary.to_csv(CSV_DIR / "cox_model_results.csv")
print(f"\nCox model concordance: {cox_model.concordance_index_:.3f}")

rhc_row = cox_model.summary.loc["rhc_Yes"]
print(
    f"RHC hazard ratio: {rhc_row['exp(coef)']:.3f} "
    f"(95% range {rhc_row['exp(coef) lower 95%']:.3f}-{rhc_row['exp(coef) upper 95%']:.3f}, "
    f"p={rhc_row['p']:.4f})"
)

top_factors = cox_model.summary.sort_values("p").head(10)
print("\nTop 10 risk factors by significance:")
print(top_factors[["exp(coef)", "p"]].to_string())

# --- Proportional hazards assumption check -----------------------------

# The Cox model assumes each covariate's hazard ratio is constant over the
# 30-day window (a patient's relative risk from a given factor doesn't
# systematically grow or shrink over time). This tests that assumption
# rather than taking it on faith; a violation doesn't invalidate the
# model's average hazard ratio, but means that ratio is a summary of a
# risk that actually shifts over the follow-up window, not a truly fixed
# value.
ph_test = proportional_hazard_test(cox_model, cox_data, time_transform="rank")
ph_test.summary.to_csv(CSV_DIR / "cox_ph_assumption_check.csv")

violations = ph_test.summary[ph_test.summary["p"] < 0.05]
print(f"\nCovariates violating the proportional hazards assumption (p < 0.05): {len(violations)} of {len(ph_test.summary)}")
if len(violations):
    print(violations[["test_statistic", "p"]].to_string())
