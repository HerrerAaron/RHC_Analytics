# RHC_Analytics
An analytics driven project that focuses on Right Heart Catherization (RHC) data. 

## Problem Statement

This project uses observational data from 5,735 ICU patients to answer the following questions:

1. **Does RHC cause a change in 30-day mortality risk (causal question)?** Right heart catheterization is an invasive procedure whose actual benefit vs. risk is disputed. Separating its true effect from the fact that sicker patients were more likely to receive it is the central question of this project.
2. **Which patient characteristics most predict mortality risk (predictive/risk-scoring question)?** Understanding what drives risk at admission is useful independent of any treatment decision, and demonstrates applied risk-scoring skills.
3. **Does the treatment effect differ across patient subgroups (segmentation question)?** An intervention that helps on average may not help, or may even harm, specific subgroups; averages alone can hide this.
4. **How does mortality risk evolve over the 30-day window, and does RHC affect timing, not just whether death occurs (survival/time-to-event question)?** Two patients who both survive or both die within 30 days can have very different trajectories; this question asks whether RHC shifts *when* death occurs, not just *whether* it occurs.

## Data Notes

Source data: `rhc.csv`, 5,735 ICU patients, 83 raw columns. Five columns had missing values; each was handled based on why it was missing, not a blanket rule.

- **dthdte (death date), 35.1% missing.** Null exactly for the 2,013 patients who survived (`death == 0`), confirmed programmatically. Structural: no death date exists because no death occurred. Left as null rather than imputed; the outcome is already captured in `death`/`death_d30`.
- **cat2 (secondary disease category), 79.1% missing.** Null for patients with only one recorded disease category, not a collection gap. Filled with the explicit label `"None"` and one-hot encoded with `None` as the reference category, instead of left as an implicit gap.
- **adld3p (ADL score), 74.9% missing.** No structural explanation, the value simply wasn't measured for most patients. Dropped: too sparse to impute reliably or trust as a feature.
- **urin1 (urine output), 52.8% missing.** Same reasoning as `adld3p`. Dropped rather than kept with a missing-indicator flag; over half the sample is missing with no structural cause.
- **dschdte (discharge date), 0.02% missing (1 patient).** Dropped along with the column itself; see below.

Additional Phase 1 decisions:

- **ptid** dropped: a redundant administrative record ID (both `patient` and `ptid` are unique across all 5,735 rows, just different numbering).
- **dschdte, lstctdte** dropped entirely, not just excluded from modeling. Neither is used anywhere in this project: the outcome derivation below only needs `sadmdte`/`dthdte`, and Phase 6's Kaplan-Meier step is built on `survial_d30`, which is already derived. Keeping unused raw dates around "just in case" isn't a real justification.
- **death, death_d30, survial_d30** recomputed from the raw admission/death dates (`sadmdte`, `dthdte`) rather than taken as given, then validated to match the provided values exactly. This also surfaced the dataset's actual censoring convention: any patient not known to have died within 30 days, whether discharged, lost to follow-up, or died after day 30, is administratively censored at exactly day 30, not their true last-contact date.
- **sadmdte, dthdte** retained as raw dates because they feed the derivation above, but excluded from any propensity-score or risk-model feature matrix, since `dthdte` is post-admission and would leak the outcome.
- **surv2md1** kept as a feature despite looking derived. It's the SUPPORT model's day-1 estimated probability of surviving 2 months, available before the RHC decision is made, and a known confounder in the original Connors et al. 1996 study.
- Dummy variables for `category`, `income`, `race`, `insurance_class`, `sex`, `cat2`, `carcinoma`, `rhc`, and `no_resus` were built from the raw columns rather than the dataset's pre-built dummy columns, which were dropped as redundant.

## Phase 2 — EDA & Naive Comparison

**EDA.** Distributions for the key admission variables (age, severity, disease category, treatment group) are in `figures/eda_key_variables.png`, with summary statistics in `eda_summary_stats.csv`. `ARenalF` (acute renal failure) is the single largest disease category at 43% of patients; the age distribution skews older, centered in the late 60s.

**Naive comparison.** Patients who received RHC had a 30-day mortality rate of 38.0%, versus 30.6% for those who didn't (`figures/naive_mortality_comparison.png`). Taken at face value, this reads as RHC increasing mortality risk, the same disputed finding that motivated the original Connors et al. 1996 study and this project's first question.

**Covariate balance.** That naive comparison isn't trustworthy on its own. Standardized mean differences (SMD) across 63 baseline covariates (`covariate_balance_naive.csv`, `figures/covariate_balance_love_plot.png`) show 34 of 63 (54%) exceed the conventional |SMD| > 0.1 imbalance threshold. Patients who received RHC were meaningfully sicker at admission: higher APACHE severity score, lower blood pressure, lower oxygenation, and more sepsis-category diagnoses than patients who didn't. The two groups are not naturally comparable, so the 38.0% vs. 30.6% gap likely reflects who was more likely to be catheterized, not the causal effect of the procedure itself. Resolving that confound is exactly what Phase 3's propensity-score adjustment addresses.
