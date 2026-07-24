"""Phase 5: re-estimate the Phase 3 IPW-adjusted treatment effect within
disease-category and severity-tier subgroups, to check whether RHC's effect
on 30-day mortality is consistent across patient segments."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Paths are anchored to this script's own location, not the working
# directory, so it runs the same regardless of where it's invoked from.
ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_DIR / "rhc_cleaned.csv")

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

# --- Propensity score and IPW weights (same model as Phase 3) ---------------

# Recomputed rather than imported: propensity_effect.py is a script with
# side effects (prints, writes files), not a reusable module. A single
# propensity model fit on the full sample is reused across all subgroups
# below, rather than refitting a 63-covariate model within each subgroup,
# which would be unstable for the smaller ones.
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

X = sm.add_constant(df[covariates])
ps_model = sm.Logit(df["rhc_Yes"], X).fit(disp=False)
df["propensity_score"] = ps_model.predict(X)

p_treated = df["rhc_Yes"].mean()
df["ipw"] = np.where(
    df["rhc_Yes"] == 1,
    p_treated / df["propensity_score"],
    (1 - p_treated) / (1 - df["propensity_score"]),
)

# --- Subgroup definitions -----------------------------------------------

# category was one-hot encoded with ARenalF as the dropped reference in
# Phase 1, so a patient with every category_* dummy at 0 is ARenalF.
category_label = pd.Series("ARenalF", index=df.index)
for c in category_cols:
    category_label[df[c] == 1] = c.replace("category_", "")
df["category_label"] = category_label

# Tertiles of aps (APACHE severity) give three roughly equal, well-powered
# groups. Disease category, in contrast, has several very small groups (see
# the sample-size filter below), which aps naturally avoids.
df["severity_tier"] = pd.qcut(
    df["aps"], q=3, labels=["Low severity", "Medium severity", "High severity"]
)

# --- Weighted treatment effect estimator (same method as Phase 3) -----------


def estimate_weighted_ate(sub_df):
    """IPW-weighted 30-day mortality risk difference (RHC vs. no RHC), with
    a 95% CI, restricted to the rows in sub_df."""
    X_outcome = sm.add_constant(sub_df["rhc_Yes"])
    model = sm.WLS(sub_df["death_d30"], X_outcome, weights=sub_df["ipw"]).fit(cov_type="HC1")
    ate = model.params["rhc_Yes"]
    ci_low, ci_high = model.conf_int().loc["rhc_Yes"]
    return ate, ci_low, ci_high


# Below this many patients in either treatment arm, a subgroup's own
# confidence interval is too unstable to report or interpret; flagged
# rather than silently estimated (see Colon cancer: 1 treated patient).
MIN_PER_ARM = 20


def subgroup_rows(group_col):
    rows = []
    for group_name, sub_df in df.groupby(group_col, observed=True):
        n_treated = int((sub_df["rhc_Yes"] == 1).sum())
        n_control = int((sub_df["rhc_Yes"] == 0).sum())
        sufficient = min(n_treated, n_control) >= MIN_PER_ARM
        ate, ci_low, ci_high = estimate_weighted_ate(sub_df) if sufficient else (np.nan, np.nan, np.nan)
        rows.append({
            "subgroup_type": group_col,
            "subgroup": group_name,
            "n": len(sub_df),
            "n_treated": n_treated,
            "n_control": n_control,
            "ate": ate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "sufficient_data": sufficient,
        })
    return rows


overall_ate, overall_ci_low, overall_ci_high = estimate_weighted_ate(df)
rows = [{
    "subgroup_type": "Overall",
    "subgroup": "All patients",
    "n": len(df),
    "n_treated": int((df["rhc_Yes"] == 1).sum()),
    "n_control": int((df["rhc_Yes"] == 0).sum()),
    "ate": overall_ate,
    "ci_low": overall_ci_low,
    "ci_high": overall_ci_high,
    "sufficient_data": True,
}]
rows += subgroup_rows("category_label")
rows += subgroup_rows("severity_tier")

subgroup_table = pd.DataFrame(rows)
subgroup_table.to_csv(CSV_DIR / "subgroup_effects.csv", index=False)
print(subgroup_table.to_string(index=False))

excluded = subgroup_table.loc[~subgroup_table["sufficient_data"], "subgroup"]
if len(excluded):
    print(f"\nExcluded for insufficient sample size (<{MIN_PER_ARM} patients in either arm): {', '.join(excluded)}")

# --- Formal interaction (effect modification) test --------------------------

# The per-subgroup estimates above only support eyeballing whether CIs
# overlap, which is not a valid test of whether subgroups differ from each
# other. This fits one pooled weighted model per subgroup type with a
# treatment x subgroup interaction, then runs a joint F-test on just the
# interaction terms: a direct test of additive-scale effect modification,
# on the same risk-difference scale as every other estimate in this project.


def interaction_test(group_col):
    """Joint F-test on the rhc_Yes x group_col interaction terms in a pooled
    weighted model. Small subgroups excluded above are dropped here too, for
    the same reason: an interaction test can't fix an unstable subgroup."""
    sufficient_groups = subgroup_table.loc[
        (subgroup_table["subgroup_type"] == group_col) & subgroup_table["sufficient_data"],
        "subgroup",
    ]
    sub = df[df[group_col].isin(sufficient_groups)]
    model = smf.wls(f"death_d30 ~ rhc_Yes * C({group_col})", data=sub, weights=sub["ipw"]).fit(cov_type="HC1")
    wald = model.wald_test_terms(skip_single=False, scalar=True)
    interaction_term = f"rhc_Yes:C({group_col})"
    return wald.table.loc[interaction_term, "statistic"], wald.table.loc[interaction_term, "pvalue"]


interaction_rows = []
for group_col in ["category_label", "severity_tier"]:
    f_stat, p_value = interaction_test(group_col)
    interaction_rows.append({"subgroup_type": group_col, "f_statistic": f_stat, "p_value": p_value})

interaction_table = pd.DataFrame(interaction_rows)
interaction_table.to_csv(CSV_DIR / "subgroup_interaction_tests.csv", index=False)
print()
print(interaction_table.to_string(index=False))

# --- Forest plot -----------------------------------------------------------

# Reversed so the table's row order reads top-to-bottom in the plot
# (matplotlib's y-axis increases upward by default).
plot_data = subgroup_table[subgroup_table["sufficient_data"]].iloc[::-1]
y_pos = range(len(plot_data))
errors = [plot_data["ate"] - plot_data["ci_low"], plot_data["ci_high"] - plot_data["ate"]]
colors = ["black" if t == "Overall" else "steelblue" for t in plot_data["subgroup_type"]]

fig, ax = plt.subplots(figsize=(7, 6))
ax.errorbar(plot_data["ate"], y_pos, xerr=errors, fmt="none", ecolor="grey", capsize=3, zorder=1)
ax.scatter(plot_data["ate"], y_pos, color=colors, zorder=2)
ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, label="No effect")
ax.axvline(overall_ate, color="firebrick", linestyle=":", linewidth=1, label="Overall effect")
ax.set_yticks(list(y_pos))
ax.set_yticklabels(plot_data["subgroup"])
ax.set_xlabel("IPW-adjusted risk difference (RHC minus no RHC)")
ax.set_title("30-day mortality effect by subgroup")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "subgroup_forest_plot.png", dpi=150)
plt.close(fig)
