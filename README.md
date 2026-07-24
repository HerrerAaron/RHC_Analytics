# RHC_Analytics
An analytics driven project that focuses on Right Heart Catherization (RHC) data. 

## Problem Statement

This project uses observational data from 5,735 ICU patients to answer the following questions:

1. **Does RHC cause a change in 30-day mortality risk (causal question)?** Right heart catheterization is an invasive procedure whose actual benefit vs. risk is disputed. Separating its true effect from the fact that sicker patients were more likely to receive it is the central question of this project.
2. **Which patient characteristics most predict mortality risk (predictive/risk-scoring question)?** Understanding what drives risk at admission is useful independent of any treatment decision, and demonstrates applied risk-scoring skills.
3. **Does the treatment effect differ across patient subgroups (segmentation question)?** An intervention that helps on average may not help, or may even harm, specific subgroups; averages alone can hide this.
4. **How does mortality risk evolve over the 30-day window, and does RHC affect timing, not just whether death occurs (survival/time-to-event question)?** Two patients who both survive or both die within 30 days can have very different trajectories; this question asks whether RHC shifts *when* death occurs, not just *whether* it occurs.

## Data Notes

Source data: `csv/rhc.csv`, 5,735 ICU patients, 83 raw columns. Five columns had missing values; each was handled based on why it was missing, not a blanket rule.

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

**EDA.** Distributions for the key admission variables (age, severity, disease category, treatment group) are in `figures/eda_key_variables.png`, with summary statistics in `csv/eda_summary_stats.csv`. `ARenalF` (acute renal failure) is the single largest disease category at 43% of patients; the age distribution skews older, centered in the late 60s.

**Naive comparison.** Patients who received RHC had a 30-day mortality rate of 38.0%, versus 30.6% for those who didn't (`figures/naive_mortality_comparison.png`). Taken at face value, this reads as RHC increasing mortality risk, the same disputed finding that motivated the original Connors et al. 1996 study and this project's first question.

**Covariate balance.** That naive comparison isn't trustworthy on its own. Standardized mean differences (SMD) across 63 baseline covariates (`csv/covariate_balance_naive.csv`, `figures/covariate_balance_love_plot.png`) show 34 of 63 (54%) exceed the conventional |SMD| > 0.1 imbalance threshold. Patients who received RHC were meaningfully sicker at admission: higher APACHE severity score, lower blood pressure, lower oxygenation, and more sepsis-category diagnoses than patients who didn't. The two groups are not naturally comparable, so the 38.0% vs. 30.6% gap likely reflects who was more likely to be catheterized, not the causal effect of the procedure itself. Resolving that confound is exactly what Phase 3's propensity-score adjustment addresses.

## Phase 3 — Causal Inference: Propensity Scores & Treatment Effect Estimation

**Propensity score model.** A logistic regression predicting `rhc_Yes` from the same 63 covariates checked in Phase 2 is in `csv/propensity_model_coefficients.csv`. Coefficients are clinically sensible: patients with more severe cardiac/sepsis presentations (`card`, `category_MOSF w/Sepsis`, `trauma`) were more likely to receive RHC, while patients with a higher SUPPORT-predicted survival probability (`surv2md1`) or a DNR order (`no_resus_Yes`) were less likely to, consistent with RHC being reserved for sicker, more aggressively treated patients. `figures/propensity_score_overlap.png` shows reasonable common support between the two groups (neither group is entirely absent from any score range), a prerequisite for the weighting approach below to be valid.

**Inverse probability weighting.** Each patient is weighted by the inverse of their estimated probability of receiving the treatment they actually received, stabilized by the overall treatment rate to control variance from patients with extreme propensity scores (weights range from 0.39 to 20.11). This reweights the sample so the RHC and no-RHC groups resemble each other on the covariates the propensity model was given, without discarding any patients.

**Post-weighting balance check.** `csv/covariate_balance_weighted.csv` and `figures/covariate_balance_before_after.png` compare each covariate's SMD before and after weighting. All 34 covariates that exceeded |SMD| > 0.1 before weighting fall under that threshold after weighting: 0 of 63 remain imbalanced. The adjustment worked, the weighted RHC and no-RHC groups are now comparable on every covariate checked.

**Adjusted treatment effect.** The naive comparison (Phase 2) showed a +7.4 percentage point gap in 30-day mortality (38.0% RHC vs. 30.6% no RHC). After IPW adjustment, estimated via weighted least squares with heteroskedasticity-robust standard errors, the effect shrinks to +5.2 percentage points (95% CI: +1.9% to +8.5%). The confidence interval excludes 0, so even after adjusting for every confounder measured here, RHC is associated with a statistically significant *increase* in 30-day mortality risk. This is observational data, though: unmeasured confounding (factors that influenced both the RHC decision and mortality but weren't recorded) can't be ruled out, so "associated with increased risk after adjustment" is not the same claim as "causes increased risk." This finding mirrors the original, controversial result from Connors et al. (1996), part of why RHC's clinical benefit remains disputed to this day.

## Phase 4 — Predictive Risk Modeling & Feature Importance

**Setup.** Three scikit-learn classifiers, logistic regression (baseline), random forest, and gradient boosting, predict `death_d30` from the same 63 admission-characteristic covariates used in Phases 2-3 (the RHC treatment indicator itself is deliberately excluded, keeping this a risk score based on admission data, not a treatment-effect model). Data is split 80/20 (stratified) into train and test sets. Model comparison, hyperparameter tuning, and model selection all happen inside stratified 5-fold cross-validation on the training set only; the held-out test set is touched exactly once, by the single winning model, for final evaluation.

**Model comparison and selection.** Each model type is hyperparameter-tuned via grid search under the same 5-fold CV (`csv/cv_model_comparison.csv`): logistic regression's best regularization strength reaches a CV ROC-AUC of 0.759, random forest 0.768, and gradient boosting 0.769. Gradient boosting is selected as the single best-performing model on CV score alone, before the test set is ever touched. On the held-out test set (`csv/test_set_metrics.csv`, `figures/roc_curves.png`), it reaches accuracy 0.742, precision 0.688, recall 0.419, and ROC-AUC 0.782, consistent with its CV score, meaning it generalizes about as well as it appeared to during cross-validation.

**Feature importance.** Permutation importance (`csv/feature_importance.csv`, `figures/feature_importance.png`), the drop in test-set ROC-AUC when a feature is randomly shuffled, is computed once, for the selected gradient boosting model. `surv2md1` (the SUPPORT model's own 2-month survival estimate) dominates by a wide margin, unsurprising since it's already a mortality-risk estimate itself. Beyond that, `das2d3pc` (functional status), `no_resus_Yes` (DNR status), and `bili1` (bilirubin) are the next-strongest drivers, a plain-language read: baseline functional status, code status, and liver function are the strongest predictors of 30-day mortality risk at admission, after the SUPPORT prognostic score itself.
