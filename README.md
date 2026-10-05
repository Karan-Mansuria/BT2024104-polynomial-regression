# BT2024104 - Polynomial Regression Assignment

Polynomial regression for the two personalized datasets in **ML-Assignment 1.pdf**.
The problems are modeled independently using fold-local preprocessing and
five-fold cross-validation for degree and regularization selection.

Repository: [Karan-Mansuria/BT2024104-polynomial-regression](https://github.com/Karan-Mansuria/BT2024104-polynomial-regression)

[Read the four-page report](output/pdf/BT2024104_report.pdf) or see the
[assignment requirements check](ASSIGNMENT_CHECKLIST.md).

## Completed results

| Problem | Selected model | Degree | Alpha | Mean selection-CV MSE | Mean selection-CV R2 |
| --- | --- | --- | --- | --- | --- |
| var1 | LASSO | 5 | 0.01 | 0.348086 | 0.966314 |
| var2 | Ridge | 12 | 1.0 | 0.235048 | 0.995178 |

Both final models converged. Each submission was independently verified to contain
1,000 finite predictions in the original test-row order. Unit tests and the
completed-results audit passed.

## Assignment scope

The PDF requires polynomial regression for two separate datasets, careful degree
selection (var1 up to 10; var2 up to 20), MSE/R2, prediction CSVs, a 4-5 page report,
and a GitHub repository containing training and inference code. It does not mandate
nested CV, three regularizers at every degree, or an exhaustive tuning grid.

This implementation uses one reproducible five-fold selection CV search. Ridge
is evaluated across every degree; LASSO and Elastic Net are compared at the two
best Ridge degrees. This is ordinary selection CV, not nested CV.

## Install and run

Use Python 3.12. Place the original four personalized CSVs and sample_submission.csv
beside this README; the CSVs must not be edited or shuffled on disk.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_assignment.py
python assignment.py train --problem both
```

On Windows use `.venv\Scripts\activate` instead. Optional arguments:
`--data-dir PATH`, `--output PATH`, `--problem var1`, or `--problem var2`.
Training runs from the beginning; there is no background supervisor, scheduler,
database, mandatory benchmark, or automatic long retry loop.
`requirements-tested.txt` records the direct numerical-library versions used for
this run; `results/environment.json` also records the Python version.

Generate CSVs from the saved fitted models without training again:

```sh
python assignment.py predict --problem both
python verify_results.py
```

Build the four-page report from completed results:

```sh
python make_report.py
```

This writes `output/pdf/BT2024104_report.pdf`. The report is generated from actual
saved metrics, explains the simplified scope, and does not claim hidden-test scores.

## Methodology

1. Load each training dataset independently; validate column names and finite
   numeric values. Record dimensions, duplicates, and training distributions.
   Duplicate input rows are retained. The target is `y`.
2. Use five shuffled folds with seed 2024104. All candidates share these folds.
3. Fit the following only on the training portion of each fold:
   raw-input standardization -> total-degree polynomial expansion (all powers and
   interactions, no constant column) -> polynomial-feature standardization ->
   regression with an intercept. Apply fitted transformations to validation rows.
4. Evaluate **Ridge at every degree 1-10 for var1 and 1-20 for var2**. One SVD per
   degree/fold provides exact predictions for all regularization strengths:
   exponents -8,-7,...,8, covering the initial -6 to 6 range and two extra decades
   on either side. Refine around each degree's best exponent at 0.25 spacing,
   bounded by [-8,8]. Reusing a fold's decomposition is a numerical optimization;
   it neither changes polynomial features nor shares validation information.
5. Shortlist the two degrees with lowest Ridge mean CV MSE. At each, compare
   LASSO and Elastic Net using descending alpha exponents 6,5,...,-8, with Elastic
   Net ratios 0.1,0.3,0.5,0.7,0.9. Warm starts are confined to a single training
   fold, degree, method, and ratio. LASSO/Elastic Net use tolerance 1e-3 and a
   maximum of 2,000 iterations, without expensive retries or further refinement.
   A candidate must converge on every fold to be eligible; failed candidates are
   logged, never silently treated as valid. Sparse-model searching is deliberately
   bounded, so the result is the best eligible tested candidate, not a claim of a
   globally optimal LASSO/Elastic Net solution.
6. Select the minimum eligible mean CV MSE across all tested candidates. Exact
   ties prefer lower degree, Ridge before LASSO before Elastic Net, larger alpha,
   then lower ratio. Compute R2 too; fold SD uses the sample standard deviation.
7. Refit the chosen pipeline separately on each fold to save OOF diagnostics,
   then on all training rows to produce the final model. Refit convergence is
   checked; a failed refit stops the run instead of emitting predictions.
8. Only now load the test data, predict in its original row order, and save one
   `y` column with no index. Reopen each CSV and independently compare batched
   predictions against the saved model. The supplied files have 1,000 rows each.

CV scores and OOF plots use folds that also influenced selection. They are useful
**selection-CV diagnostics**, but are not unbiased nested-CV or hidden-test scores.
Test targets are unavailable. No test values, ranges, or distributions influence
model selection or preprocessing choices. A selected alpha at an absolute search
boundary is flagged in selection.json and should be reported as unresolved.
The LASSO/Elastic Net search uses warm starts, while OOF refits start afresh. At the
specified numerical tolerance their predictions can differ slightly; both the
search MSE and independently refitted OOF MSE are recorded in selection.json.

## Main files

- `assignment.py`: training, final refitting, inference, and CSV verification.
- `search.py`: all Ridge, LASSO, Elastic Net, preprocessing, and search logic.
- `reporting.py`: comparison tables and report plots.
- `make_report.py`: reproducible four-page PDF report from completed results.
- `verify_results.py`: independent search-coverage, model-selection and CSV audit.
- `test_assignment.py`: exact Ridge equivalence, scaling isolation, and convergence tests.
- `requirements.txt`: runtime dependencies.

## Outputs

The two submission files are `results/BT2024104_pred_var1.csv` and
`results/BT2024104_pred_var2.csv`. Each problem's folder also contains:

- `selection.json`: selected degree, method, alpha, CV MSE/R2, and convergence.
- `model.joblib`: complete fitted preprocessing-and-regression pipeline.
- `fold_results.csv`, `cv_results.csv`: all tested settings and fold/summary metrics.
- `cv_splits.csv`, `selected_fold_metrics.csv`, `oof_predictions.csv`.
- `degree_method_summary.csv`, `degree_comparison.png`, `oof_diagnostics.png`.
- `audit.json`, `training_summary.csv`, `ridge_runtime.csv`.
- `prediction_verification.json`: file shape, saved-model consistency, and SHA256.

Ridge timing is measured per degree including decomposition; its candidate rows
have zero fit_seconds because that work is shared. Sparse fit_seconds is actual
per-fit solver time. The two timing fields should not be compared as independent
candidate runtimes. Only load model.joblib files produced by your own trusted run.

The repository contains the training and inference code, dependencies, tests and
report. Personalized source datasets, old experiments, local environments and
generated training outputs are excluded by .gitignore. Supply the four original
BT2024104 data files locally before running training or inference. The final
submission bundle separately contains both prediction CSVs and the fitted models.
