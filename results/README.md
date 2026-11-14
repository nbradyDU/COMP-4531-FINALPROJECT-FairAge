# FairAge results

The public notebooks read nine CSVs from [`published/`](published/). Other results stay local.

| Files | Contents |
|---|---|
| [Models](published/models.csv), [histories](published/histories.csv) | Settings, selected epochs, and BF16 training histories |
| [Evaluations](published/evaluations.csv), [per age](published/per_age.csv), [confusion](published/confusion.csv) | FP32 development and holdout scores, confidence, and errors |
| [Age by race](published/age_race.csv) | Image counts and accuracy by age and race |
| [Studies](published/optuna_studies.csv), [trials](published/optuna_trials.csv) | Optuna settings, trial states, and BF16 objectives |
| [Dataset counts](published/dataset_counts.csv) | Publisher training counts before exclusions and study training counts after exclusions |

In `age_race.csv`, cells with fewer than 30 images retain their counts and mark accuracy as `suppressed`. A missing score is unreported. Counts alone cannot establish reliability.

Confidence summaries follow the publication producer at commit `8f11582`. Confidence is the maximum of nine class probabilities. ECE uses 10 or 15 equal-width bins on [0,1], with each lower edge included, each upper edge excluded, and 1 included in the final bin. It is the sample-weighted absolute gap between accuracy and mean confidence in each nonempty bin. `wrong_above_90` counts incorrect predictions with confidence strictly greater than 0.9; `wrong_count` is the denominator for their fraction. Explicit FP32 evaluation now saves these summaries in `metrics.json`; the published CSVs remain fixed snapshots.

Open the [results notebook](../notebooks/03_results_analysis.ipynb) to explore the tables.
