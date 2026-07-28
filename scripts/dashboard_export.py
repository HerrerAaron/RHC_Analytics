"""Phase 7 data prep: export clean, Power BI-ready CSVs for the dashboard's
five panels (population overview, naive vs. adjusted effect, subgroup
comparison, feature importance, survival curves). Reuses validated values
from earlier phases rather than recomputing cleaning/modeling logic; the
only new work here is reshaping for a BI tool instead of for modeling."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "csv"

df = pd.read_csv(CSV_DIR / "rhc_cleaned.csv")

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

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

# --- Population overview -----------------------------------------------

# One-hot dummies are correct for modeling but unusable for a Power BI
# slicer, a stakeholder wants one "Disease Category" dropdown, not eight
# separate 0/1 columns. This collapses a family of prefix_* dummy columns
# (with `baseline` as the dropped Phase 1 reference level) back into a
# single readable text column.


def reconstruct_category(prefix, baseline):
    cols = [c for c in df.columns if c.startswith(prefix)]
    label = pd.Series(baseline, index=df.index)
    for c in cols:
        label[df[c] == 1] = c[len(prefix):]
    return label


population = pd.DataFrame({
    "patient": df["patient"],
    "age": df["age"],
    "sex": np.where(df["sex_Male"] == 1, "Male", "Female"),
    "race": reconstruct_category("race_", "White"),
    "income": reconstruct_category("income_", "Under $11k"),
    "insurance_class": reconstruct_category("insurance_class_", "Private"),
    "disease_category": reconstruct_category("category_", "ARenalF"),
    "secondary_diagnosis": reconstruct_category("cat2_", "None"),
    "carcinoma": reconstruct_category("carcinoma_", "No"),
    "severity_aps": df["aps"],
    "severity_scoma1": df["scoma1"],
    "no_resuscitate_order": np.where(df["no_resus_Yes"] == 1, "Yes", "No"),
    "received_rhc": np.where(df["rhc_Yes"] == 1, "Yes", "No"),
    "died_within_30_days": np.where(df["death_d30"] == 1, "Yes", "No"),
})
population.to_csv(CSV_DIR / "dashboard_population.csv", index=False)

# --- Naive vs. adjusted treatment effect ------------------------------------

# Same propensity model and IPW weights as Phase 3, recomputed here rather
# than imported since propensity_effect.py is a script, not a module.
X = sm.add_constant(df[covariates])
ps_model = sm.Logit(df["rhc_Yes"], X).fit(disp=False)
df["propensity_score"] = ps_model.predict(X)

p_treated = df["rhc_Yes"].mean()
df["ipw"] = np.where(
    df["rhc_Yes"] == 1,
    p_treated / df["propensity_score"],
    (1 - p_treated) / (1 - df["propensity_score"]),
)

treated_mask = df["rhc_Yes"] == 1

naive_rhc = df.loc[treated_mask, "death_d30"].mean()
naive_no_rhc = df.loc[~treated_mask, "death_d30"].mean()

# Weighted group means, the same reweighting Phase 3 used to compute the
# adjusted risk difference, but reported per arm instead of as a single
# subtracted number, so the dashboard can show two comparable bars.
adjusted_rhc = np.average(df.loc[treated_mask, "death_d30"], weights=df.loc[treated_mask, "ipw"])
adjusted_no_rhc = np.average(df.loc[~treated_mask, "death_d30"], weights=df.loc[~treated_mask, "ipw"])

X_outcome = sm.add_constant(df["rhc_Yes"])
outcome_model = sm.WLS(df["death_d30"], X_outcome, weights=df["ipw"]).fit(cov_type="HC1")
adjusted_gap = outcome_model.params["rhc_Yes"]
ci_low, ci_high = outcome_model.conf_int().loc["rhc_Yes"]

naive_vs_adjusted = pd.DataFrame([
    {
        "comparison": "Naive (unadjusted)",
        "rhc_mortality_rate": naive_rhc,
        "no_rhc_mortality_rate": naive_no_rhc,
        "risk_difference": naive_rhc - naive_no_rhc,
        "ci_low": np.nan,
        "ci_high": np.nan,
    },
    {
        "comparison": "Adjusted (IPW)",
        "rhc_mortality_rate": adjusted_rhc,
        "no_rhc_mortality_rate": adjusted_no_rhc,
        "risk_difference": adjusted_gap,
        "ci_low": ci_low,
        "ci_high": ci_high,
    },
])
naive_vs_adjusted.to_csv(CSV_DIR / "dashboard_naive_vs_adjusted.csv", index=False)

# --- Kaplan-Meier curve data -------------------------------------------

treated = df[df["rhc_Yes"] == 1]
control = df[df["rhc_Yes"] == 0]

km_treated = KaplanMeierFitter(label="RHC")
km_treated.fit(treated["survial_d30"], event_observed=treated["death_d30"])

km_control = KaplanMeierFitter(label="No RHC")
km_control.fit(control["survial_d30"], event_observed=control["death_d30"])


def km_to_long(fitter, group_name):
    """Reshape a fitted KaplanMeierFitter's step function and confidence
    band into a tidy (day, survival_probability, ci_low, ci_high, group)
    table, the format a BI tool's line chart needs."""
    sf = fitter.survival_function_.reset_index()
    sf.columns = ["day", "survival_probability"]
    ci = fitter.confidence_interval_.reset_index()
    ci.columns = ["day", "ci_low", "ci_high"]
    merged = sf.merge(ci, on="day")
    merged["group"] = group_name
    return merged


km_curve_data = pd.concat([
    km_to_long(km_treated, "RHC"),
    km_to_long(km_control, "No RHC"),
], ignore_index=True)
km_curve_data.to_csv(CSV_DIR / "dashboard_km_curve.csv", index=False)

print(f"Population overview: {len(population)} rows -> dashboard_population.csv")
print(f"Naive vs. adjusted: {len(naive_vs_adjusted)} rows -> dashboard_naive_vs_adjusted.csv")
print(f"KM curve data: {len(km_curve_data)} rows -> dashboard_km_curve.csv")
