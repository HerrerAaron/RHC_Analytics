"""Phase 1 data prep: load rhc.csv, resolve missingness, derive outcome
columns, and one-hot encode categoricals."""

import pandas as pd

# Headers and string values are whitespace-padded in the source CSV
# (e.g. "NA      "), so na_values=["NA"] misses them on a naive read.
df = pd.read_csv("rhc.csv", skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]

str_cols = df.select_dtypes(include=["object", "str"]).columns
for c in str_cols:
    df[c] = df[c].str.strip()
df = df.replace("NA", pd.NA)

# --- Missingness -------------------------------------------------------

# dthdte null iff death == 0 (structural: survivors have no death date).
# Left null; outcome is already captured in death/death_d30.
assert (df["dthdte"].isna() == (df["death"] == 0)).all()

# cat2 null means "no secondary diagnosis," not a collection gap.
df["cat2"] = df["cat2"].fillna("None")

# adld3p (74.9% missing) and urin1 (52.8% missing): no structural cause,
# too sparse to impute. ptid: redundant admin ID (unique 1:1 with patient).
# dschdte, lstctdte: unused by any derivation below or planned downstream
# step (Phase 6 uses survial_d30, already derived from sadmdte/dthdte).
df = df.drop(columns=["adld3p", "urin1", "ptid", "dschdte", "lstctdte"])

# sadmdte, dthdte: retained as raw dates because they feed the outcome
# derivation below. Exclude from any propensity/risk feature matrix
# regardless: dthdte is post-admission and would leak the outcome.
#
# surv2md1 is not administrative: SUPPORT model's day-1 estimated 2-month
# survival probability, available pre-treatment. Retained as a feature.

# --- Derived outcome columns --------------------------------------------

# death, death_d30, survial_d30 ship pre-computed. Rebuilt from
# sadmdte/dthdte and validated against the provided values below.
#
# Censoring convention (per the asserts): a patient not known to have died
# within 30 days, whether discharged, lost to follow-up, or died after day
# 30, is right censored at exactly day 30, not their true last-contact date.
df["sadmdte"] = pd.to_numeric(df["sadmdte"])
df["dthdte"] = pd.to_numeric(df["dthdte"])

days_to_death = df["dthdte"] - df["sadmdte"]
died_by_30 = df["dthdte"].notna() & (days_to_death <= 30)

my_death = df["dthdte"].notna().astype(int)
my_death_d30 = died_by_30.astype(int)
my_survial_d30 = days_to_death.where(died_by_30, other=30)

assert (my_death == df["death"]).all()
assert (my_death_d30 == df["death_d30"]).all()
assert (my_survial_d30 == df["survial_d30"]).all()

df["death"] = my_death
df["death_d30"] = my_death_d30
df["survial_d30"] = my_survial_d30

# --- Categorical encoding ------------------------------------------------

# category, income, race, insurance_class, sex: no missing values. Encoded
# from raw columns rather than the dataset's pre-built dummies, which are
# dropped below as redundant.
#
# category/income/race/insurance_class get an explicit reference category
# instead of drop_first's default alphabetical pick, which produced
# non-obvious baselines (income's reference was "$11-$25k", an artifact of
# "$" sorting before "U", not the lowest bracket). Pinned to the most
# common or most conventional level per variable: ARenalF (largest disease
# category, 43% of patients), Under $11k (lowest income bracket), White
# (largest race group), Private (largest insurance group, standard
# reference in health insurance research).
category_categories = ["ARenalF"] + sorted(v for v in df["category"].unique() if v != "ARenalF")
df["category"] = pd.Categorical(df["category"], categories=category_categories)

income_categories = ["Under $11k"] + sorted(v for v in df["income"].unique() if v != "Under $11k")
df["income"] = pd.Categorical(df["income"], categories=income_categories)

race_categories = ["White"] + sorted(v for v in df["race"].unique() if v != "White")
df["race"] = pd.Categorical(df["race"], categories=race_categories)

insurance_categories = ["Private"] + sorted(v for v in df["insurance_class"].unique() if v != "Private")
df["insurance_class"] = pd.Categorical(df["insurance_class"], categories=insurance_categories)

dummy_source_cols = ["category", "income", "race", "insurance_class", "sex"]
df = pd.get_dummies(df, columns=dummy_source_cols, drop_first=True)

prebuilt_dummy_cols = [
    "inc_2", "inc_3", "inc_4",
    "dis_2", "dis_3", "dis_4", "dis_5", "dis_6", "dis_7", "dis_8", "dis_9",
    "race1", "race2",
    "insur_1", "insure_2", "insur_3", "insur_4", "insur_5",
]
df = df.drop(columns=prebuilt_dummy_cols)

# cat2, carcinoma, rhc, no_resus: same treatment, not named in the roadmap.
# rhc/no_resus are plain Yes/No, so alphabetical drop_first ("No" dropped)
# is already correct. cat2/carcinoma get their "no condition" level pinned
# as the explicit reference so the dropped baseline is meaningful, not an
# arbitrary alphabetically-first disease name.
cat2_categories = ["None"] + sorted(v for v in df["cat2"].unique() if v != "None")
df["cat2"] = pd.Categorical(df["cat2"], categories=cat2_categories)

carcinoma_categories = ["No"] + sorted(v for v in df["carcinoma"].unique() if v != "No")
df["carcinoma"] = pd.Categorical(df["carcinoma"], categories=carcinoma_categories)

more_dummy_cols = ["cat2", "carcinoma", "rhc", "no_resus"]
df = pd.get_dummies(df, columns=more_dummy_cols, drop_first=True)

# ca_1/ca_2: pre-built carcinoma dummies, redundant now.
df = df.drop(columns=["ca_1", "ca_2"])

# print(f"columns: {df.shape[1]}, rows: {df.shape[0]}")
# print(df.isna().sum().loc[lambda s: s > 0])

# --- Export ---------------------------------------------------------------
df.to_csv("rhc_cleaned.csv", index=False)
