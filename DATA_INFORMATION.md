# MS Sentinel - Data Information

## 1. Project Purpose

This project uses longitudinal real-world multiple sclerosis clinical data.

Primary modeling objective:

Predict whether a patient will experience confirmed disability progression during the next 24 months using only information available on or before a prediction/index date.

The first version should focus on EDSS-based disability progression.

Do NOT call the outcome PIRA yet because the available relapse table represents relapse-treatment episodes and may not contain every confirmed relapse.

The real patient data will remain local. The code must be written so that the user can provide the real data directory at runtime.

Example:

python scripts/audit_data.py --data-dir /path/to/processed


## 2. Privacy and Data Handling

The processed dataset is pseudonymized confidential health data.

Rules:

- Never search for or access raw identifiable data.
- Never access the patient ID mapping file.
- Never upload patient data.
- Never send patient-level data to APIs or external services.
- Never use cloud experiment tracking.
- Never commit CSV/Parquet patient data to Git.
- Do not print patient-level rows into logs.
- Aggregate statistics are allowed.
- Use only `study_id` as the patient identifier.
- All analysis and model training will run locally.
- No external telemetry involving patient data.
- No external LLM/API calls using patient data.


## 3. Dataset Overview

There are approximately 2,658 unique patients.

Available processed tables:

| File | Approx. rows | Approx. patients |
|---|---:|---:|
| patients.csv | 2,658 | 2,658 |
| edss.csv | 30,516 | 2,657 |
| clinical_tests.csv | 8,135 | 1,353 |
| treatments.csv | 4,826 | 1,948 |
| relapse_therapy.csv | 4,415 | 1,486 |
| csf.csv | 1,871 | 1,820 |
| sdmt.csv | 11,432 | 1,811 |
| vas_qol.csv | 7,356 | 1,703 |
| eq5d.csv | 5,604 | 1,224 |

The main strength of the dataset is its longitudinal EDSS history.

EDSS coverage:

- 2,596 patients have >=2 EDSS measurements
- 2,437 have >=3
- 2,026 have >=5
- 1,212 have >=10
- median measurements per patient: 9
- median EDSS follow-up: approximately 11.8 years
- 2,101 patients have >=5 years EDSS follow-up


# 4. patients.csv

## Meaning

One row per patient.

Contains demographic and high-level disease information.

## Columns

| Column | Expected type | Meaning / notes |
|---|---|---|
| study_id | string | pseudonymized patient identifier |
| date_of_birth | datetime | do not use directly as model feature |
| country_of_birth | categorical/string | optional, not initially required |
| sex | categorical | usually M/F |
| first_exam_date | datetime | first recorded examination at center |
| ms_onset_date | datetime | estimated MS onset |
| age_at_onset | float | age at MS onset |
| origin | categorical/string | optional |
| education | categorical/string | low/middle/high etc. |
| employment_status | categorical/string | optional secondary variable |
| marital_status | categorical/string | optional secondary variable |
| clinical_phase_date | datetime | date associated with clinical phase |
| clinical_phase | categorical/string | e.g. relapse-remitting/progressive |
| smoking_status | categorical/string | yes/no/unknown |
| deceased | boolean/int | do not use as predictor initially |
| date_deceased | datetime | do not use as predictor |
| birth_year | integer | year of birth |

At each prediction/index date derive:

- age_at_index
- disease_duration_years

Do not use future clinical-phase information.

Do not use deceased or date_deceased as predictors.


# 5. edss.csv

## Meaning

Longitudinal Expanded Disability Status Scale measurements.

This is the main clinical trajectory table and main source for the progression outcome.

## Columns

| Column | Type |
|---|---|
| study_id | string |
| date | datetime |
| edss | float |

Expected EDSS range: 0 to 10.

Approximate records:

30,516 observations across 2,657 patients.

Approximate date range:

1959 to 2024.

Important derived features may include:

- current_edss
- previous_edss
- edss_change_6m
- edss_change_12m
- edss_change_24m
- edss_slope_12m
- edss_slope_24m
- edss_mean_24m
- edss_min_24m
- edss_max_24m
- edss_std_24m
- edss_count_24m
- days_since_previous_edss


# 6. clinical_tests.csv

## Meaning

Repeated functional and cognitive clinical tests.

## Columns

| Column | Type | Meaning |
|---|---|---|
| study_id | string | patient |
| date | datetime | test date |
| t25fw_trial1 | float | Timed 25-Foot Walk trial 1, seconds |
| t25fw_trial2 | float | trial 2 |
| t25fw_mean | float | mean walking time |
| nine_hpt_right_trial1 | float | Nine-Hole Peg Test right hand, seconds |
| nine_hpt_right_trial2 | float | second right-hand trial |
| nine_hpt_right_mean | float | mean right-hand time |
| nine_hpt_left_trial1 | float | left-hand trial |
| nine_hpt_left_trial2 | float | second left-hand trial |
| nine_hpt_left_mean | float | mean left-hand time |
| dominant_hand | categorical/integer | coding unknown until verified |
| pasat3 | float | PASAT cognitive score |

Approximately:

8,135 records across 1,353 patients.

Important issue:

Some records contain values such as T25FW = 0 or PASAT = 0.

Do not automatically assume zero is a valid measurement.

For tests where zero is physiologically impossible, such as walking time, treat zero as suspicious and quantify it before deciding whether it represents missing data.

For PASAT, determine whether zero is a valid score or historical missing-data encoding before modifying it.


# 7. sdmt.csv

## Meaning

Longitudinal Symbol Digit Modalities Test measurements.

Used to represent cognitive processing speed.

## Columns

| Column | Type |
|---|---|
| study_id | string |
| date | datetime |
| sdmt | float |

Approximately:

11,432 measurements across 1,811 patients.

Possible features:

- latest_sdmt
- sdmt_change_12m
- sdmt_change_24m
- sdmt_slope_12m
- sdmt_slope_24m
- sdmt_count_24m
- days_since_last_sdmt

Missing SDMT must remain missing.

Do not assume absence of SDMT represents cognitive impairment.


# 8. treatments.csv

## Meaning

Longitudinal disease-modifying treatment episodes.

One row represents one treatment episode.

## Columns

| Column | Type |
|---|---|
| study_id | string |
| treatment_id | string |
| start_date | datetime |
| latest_date | datetime |
| end_date | datetime |
| status | categorical/string |
| inn | categorical/string |
| trade_name | categorical/string |
| cumulative_dose | float |
| reason_for_stopping | categorical/free text |

Approximate records:

4,826 episodes across 1,948 patients.

The table contains both older and modern MS treatments.

Examples may include:

- interferon beta
- glatiramer acetate
- natalizumab
- ocrelizumab
- fingolimod
- cyclophosphamide
- mitoxantrone
- alemtuzumab
- stem-cell transplantation

Possible derived features:

- currently_on_dmt
- current_dmt
- current_dmt_duration_days
- number_previous_dmts
- number_total_switches
- switches_previous_24m
- days_since_last_switch
- previous_dmt
- recent_discontinuation

`treatment_id` is administrative and must not be used as a predictor.

CRITICAL:

reason_for_stopping can only be used if the treatment ended before the prediction/index date.

A future reason such as "disease progression" or "inefficacy" would cause severe target leakage.


# 9. relapse_therapy.csv

## Meaning

Episodes of relapse treatment or other acute MS treatment.

This table should NOT automatically be interpreted as a complete confirmed-relapse registry.

## Columns

| Column | Type |
|---|---|
| study_id | string |
| type | categorical |
| start_date | datetime |
| end_date | datetime |
| status | categorical |
| inn | categorical |
| trade_name | categorical |
| technique | categorical/string |
| indication | categorical/string |
| route | categorical |
| posology | numeric/string |
| unit | categorical |
| duration | float |
| corticosteroid | boolean/int |

Approximately:

4,415 episodes across 1,486 patients.

Many episodes contain methylprednisolone / Solumedrol treatment.

Use terminology:

"relapse-treatment episode"

rather than:

"confirmed relapse"

Possible derived features:

- relapse_therapy_count_6m
- relapse_therapy_count_12m
- relapse_therapy_count_24m
- corticosteroid_episode_count_12m
- corticosteroid_episode_count_24m
- days_since_last_relapse_therapy

Do not implement true PIRA using this table unless relapse completeness is later validated.


# 10. csf.csv

## Meaning

CSF and associated laboratory information.

Most CSF measurements are likely diagnostic/baseline information rather than dense longitudinal progression markers.

## Columns

| Column | Type |
|---|---|
| study_id | string |
| date | datetime |
| blood_contamination | categorical/string |
| white_cell_count | float |
| lymphocytes_pct | float |
| monocytes_pct | float |
| neutrophils_pct | float |
| total_proteins | float |
| csf_albumin | float |
| blood_albumin | float |
| csf_igg | float |
| blood_igg | float |
| igg_index | float |
| igg_index_abnormal | categorical |
| oligoclonal_bands | categorical |

Expected categorical values may include:

igg_index_abnormal:
- Normal
- Abnormal
- Unknown

oligoclonal_bands:
- Yes
- No
- Unknown

Approximately:

1,871 observations across 1,820 patients.

There is substantial missingness within individual laboratory fields.

CSF should not be part of the first core model.

It may later be tested as an optional baseline feature group.


# 11. vas_qol.csv

## Meaning

Longitudinal Visual Analogue Scale quality-of-life measurements.

## Columns

| Column | Type |
|---|---|
| study_id | string |
| date | datetime |
| vas_qol | float |

Approximately:

7,356 observations across 1,703 patients.

Likely expected range: 0-100.

Validate this empirically before filtering.

Potential features:

- latest_vas_qol
- vas_change_12m
- vas_change_24m
- vas_slope_24m

Initially treat QoL as a secondary feature group.


# 12. eq5d.csv

## Meaning

Longitudinal EQ-5D health-state measurements.

## Columns

| Column | Type |
|---|---|
| study_id | string |
| date | datetime |
| raw_value | string |
| mobility | integer |
| self_care | integer |
| usual_activities | integer |
| pain_discomfort | integer |
| anxiety_depression | integer |

Examples of raw values may look like:

2-1-2-2-1

Do not calculate a utility score until the exact EQ-5D version and appropriate value set have been established.

For now use the individual dimensions only.


# 13. Date Handling

Dates may appear in CSV files in forms such as:

1/31/13
10/26/20
8/20/12

Load all date columns using robust datetime parsing.

Convert internally to a consistent datetime representation.

Do not silently discard unparseable dates.

Generate a report of:

- invalid dates
- ambiguous dates
- dates before birth
- MS onset before birth
- treatment end before treatment start


# 14. Missing Values

Never automatically convert missing values to zero.

NaN means missing unless documentation proves otherwise.

Before modeling generate:

- missingness by variable
- missingness by table
- missingness by patient
- missingness by calendar year

All statistical imputation must be fitted using training data only.

Potential baseline approaches:

numeric:
median imputation

categorical:
explicit "Missing/Unknown" category

XGBoost may use native missing-value handling.


# 15. Main Prediction Design

Prediction unit:

one patient at one historical index date.

Candidate index dates should initially be EDSS measurement dates.

For each prediction snapshot:

Historical/lookback window:
24 months

Prediction horizon:
24 months

Minimum spacing between index dates:
12 months

All parameters should be configurable.


# 16. Outcome

Primary outcome:

24-month confirmed EDSS disability progression.

The exact threshold must NOT be hard-coded throughout the project.

Create a configuration file such as:

configs/outcome.yaml

The progression rule should support baseline-dependent EDSS worsening thresholds and configurable confirmation windows.

The final clinical threshold definition will later be confirmed with an MS clinical collaborator.

Positive outcome should require:

EDSS worsening + confirmation on a later assessment

rather than one isolated EDSS increase.


# 17. Follow-up / Censoring

Do not treat:

"no recorded progression"

as:

"no progression"

when there is insufficient follow-up.

Patients without adequate observation during the future prediction window should be marked:

unknown/censored

or excluded according to the predefined cohort rules.

The cohort report must show:

- total candidate snapshots
- eligible snapshots
- positive outcomes
- negative outcomes
- censored/unknown snapshots
- unique patients represented


# 18. Leakage Prevention

At index date T:

Every model feature must satisfy:

feature_date <= T

Information after T may only be used to determine the outcome.

Never use as predictors:

- Mean ARMSS
- Mean MSSS
- Last available clinical follow-up
- Most recent info
- date_deceased
- deceased
- future clinical phase
- future EDSS
- future SDMT
- future functional tests
- future treatment changes
- future treatment stopping reason
- future relapse therapy

unless explicitly required to determine an outcome.


# 19. Initial Feature Groups

Model A:
- age_at_index
- sex
- disease_duration_years
- current_edss

Model B:
Model A +
longitudinal EDSS features

Model C:
Model B +
DMT/treatment history +
relapse-treatment history

Model D:
Model C +
SDMT +
T25FW +
9HPT +
PASAT

Later Model E:
Model D +
quantitative MRI biomarkers


# 20. Initial Algorithms

Start with:

1. simple current-EDSS baseline
2. Logistic Regression
3. Random Forest
4. XGBoost

Do not start with deep learning.

Deep learning/temporal models should only be considered after the classical pipeline is correct and clinically meaningful.


# 21. Data Splitting

CRITICAL:

Split by `study_id`.

Never randomly split individual prediction rows.

Every snapshot belonging to a particular patient must remain in exactly one dataset.

Suggested initial split:

70% patients training
15% patients validation
15% patients testing

Use reproducible random seeds.

Later add temporal validation.


# 22. Evaluation

Report at minimum:

- AUROC
- AUPRC
- sensitivity
- specificity
- PPV
- NPV
- F1 score
- Brier score

Also produce:

- ROC curve
- precision-recall curve
- calibration curve
- confusion matrix

Use patient-level bootstrap confidence intervals.

Also report clinically useful alert metrics such as:

- proportion of patients/snapshots flagged
- number of alerts required to identify one true progression case


# 23. Explainability

Use SHAP for the best tree-based model.

Generate:

- global SHAP summary
- feature importance
- selected dependence plots

Patient-level explanations may show:

- estimated progression risk
- features pushing risk upward
- features pushing risk downward

Do not describe SHAP features as causal effects.


# 24. Required Data Quality Checks

Before any model training check:

- duplicate rows
- duplicate patient/date observations
- EDSS outside 0-10
- impossible dates
- visit before birth
- MS onset before birth
- treatment end before start
- negative treatment durations
- overlapping treatment episodes
- impossible T25FW values
- impossible 9HPT values
- suspicious zero values
- SDMT distribution
- PASAT distribution
- VAS QoL distribution
- invalid EQ-5D dimensions
- unknown categorical values
- missingness patterns
- calendar-year data availability

The data spans several decades, so clinical practice and test availability have changed substantially over time.

Generate plots/tables for:

- patients per year
- EDSS measurements per year
- SDMT measurements per year
- functional tests per year
- DMT use over time
- relapse-treatment episodes over time


# 25. Pipeline Order

Do NOT immediately train the final ML models.

First complete:

1. schema validation
2. data-quality audit
3. missingness audit
4. candidate index-date generation
5. outcome construction
6. censoring rules
7. leakage tests
8. feature generation
9. cohort summary

At that point generate an aggregate cohort report and stop for review.

Only after the cohort is confirmed should model training proceed.


# 26. MRI

Quantitative MRI data exist separately but are not yet part of this processed dataset.

MRI will be integrated after the clinical-only pipeline is validated.

Expected future MRI information may include:

- study date
- scanner
- field strength
- intracranial volume
- whole-brain volume
- gray-matter volume
- white-matter volume
- CSF volume
- FLAIR lesion volume
- T1 black-hole volume
- head-size normalization
- spinal-cord C2:C3 area

The project architecture should therefore allow a later MRI feature module without redesigning the pipeline.


# 27. Expected Execution

Code should accept the real data path at runtime.

Example:

python scripts/audit_data.py \
    --data-dir /Users/.../processed

python scripts/build_cohort.py \
    --data-dir /Users/.../processed

python scripts/train_models.py \
    --data-dir /Users/.../processed

Do not hard-code a user's absolute local path.


# 28. Important Scientific Principle

The goal is not simply to achieve the highest AUROC.

The pipeline must answer:

1. Can future disability progression be predicted from routine clinical data?
2. Does longitudinal EDSS improve prediction over current EDSS alone?
3. Does treatment/relapse history add predictive information?
4. Do cognitive and functional measures add further predictive information?
5. Later, does quantitative MRI improve prediction beyond clinical data?

Every modeling decision must prioritize leakage prevention, reproducibility, calibration, interpretability, and clinical validity.