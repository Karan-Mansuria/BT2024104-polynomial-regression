# Assignment requirements check

Checked against all three pages of `ML-Assignment 1.pdf`.

| Requirement | Evidence |
| --- | --- |
| Use the personalized BT2024104 datasets | Input filenames and SHA256 checks in `audit.json` |
| Treat var1 and var2 as separate problems | Independent search, fitted pipeline and prediction file per problem |
| Only polynomial regression | Polynomial features followed by Ridge, LASSO or Elastic Net; no other predictor |
| var1: six inputs and degree at most 10 | Ridge degrees 1-10 evaluated; final degree 5 |
| var2: three inputs and degree at most 20 | Ridge degrees 1-20 evaluated; final degree 12 |
| Degree means total power across each term | `PolynomialFeatures(include_bias=False)` includes powers and interactions of total degree at most d |
| Carefully choose the polynomial degree | Shared five-fold validation splits, candidate MSE comparison and degree plots |
| Evaluate MSE and R2 | Fold-wise and summary metrics, report tables; no unsupported hidden-test score |
| Report of at most 4-5 pages | Four-page `output/pdf/BT2024104_report.pdf` |
| Explain approach, selected degrees, rationale and other techniques | Report describes scaling, regularization, CV, degree curves, solver choices and limitations |
| Two prediction CSVs with required filenames | `results/BT2024104_pred_var1.csv` and `results/BT2024104_pred_var2.csv` |
| Match sample-submission format | One `y` column, no index, 1,000 finite predictions per file; original test order preserved |
| GitHub repository containing training and inference code | [Karan-Mansuria/BT2024104-polynomial-regression](https://github.com/Karan-Mansuria/BT2024104-polynomial-regression) |

The PDF does not require nested CV, gradient descent, a particular solver,
an iteration count, or comparisons of every regularizer at every degree.
These are implementation choices, not missing PDF requirements.
