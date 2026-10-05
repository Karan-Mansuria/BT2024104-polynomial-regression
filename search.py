"""Practical polynomial regression search; all transformations are fold-local.

Ridge uses one SVD per fold/degree for an entire alpha grid. This is exactly
the Ridge solution, not a different model or an approximation to the features.
LASSO and Elastic Net use descending, warm-started alpha paths on two degrees.
"""
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass(frozen=True)
class Settings:
    seed: int = 2024104
    folds: int = 5
    shortlist: int = 2
    tolerance: float = 1e-3
    max_iter: int = 2000
    ratios: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)


def preprocessing(degree):
    return Pipeline([
        ('raw_scaler', StandardScaler()),
        ('polynomial', PolynomialFeatures(degree=degree, include_bias=False)),
        ('polynomial_scaler', StandardScaler()),
    ])


def make_model(method, alpha, ratio, settings):
    if method == 'Ridge':
        return Ridge(alpha=alpha, solver='svd')
    options = dict(alpha=alpha, tol=settings.tolerance, max_iter=settings.max_iter,
                   selection='cyclic', warm_start=True)
    if method == 'LASSO':
        return Lasso(**options)
    if method == 'ElasticNet':
        return ElasticNet(l1_ratio=ratio, **options)
    raise ValueError(f'Unknown method: {method}')


def make_pipeline(degree, method, alpha, ratio, settings):
    return Pipeline(preprocessing(degree).steps + [
        ('regression', make_model(method, alpha, ratio, settings))])


def metric_row(degree, method, alpha, ratio, fold, actual, predicted,
               converged=True, iterations=0, gap=0.0, seconds=0.0):
    return dict(degree=int(degree), method=method, alpha=float(alpha),
                l1_ratio=float(ratio), fold=int(fold),
                mse=float(mean_squared_error(actual, predicted)),
                r2=float(r2_score(actual, predicted)), converged=bool(converged),
                iterations=int(iterations), dual_gap=float(gap), seconds=float(seconds))


def summarize(rows):
    frame = pd.DataFrame(rows)
    result = frame.groupby(['degree', 'method', 'alpha', 'l1_ratio'], as_index=False).agg(
        mean_mse=('mse', 'mean'), std_mse=('mse', 'std'),
        mean_r2=('r2', 'mean'), std_r2=('r2', 'std'),
        converged_folds=('converged', 'sum'), folds=('fold', 'count'),
        fit_seconds=('seconds', 'sum'), max_iterations=('iterations', 'max'))
    result['eligible'] = (result.converged_folds == result.folds)
    return result


def rank(frame):
    valid = frame.loc[frame.eligible].copy()
    valid['method_order'] = valid.method.map({'Ridge': 0, 'LASSO': 1, 'ElasticNet': 2})
    return valid.sort_values(['mean_mse', 'degree', 'method_order', 'alpha', 'l1_ratio'],
                             ascending=[True, True, True, False, True])


def ridge_search(X, y, splits, maximum_degree):
    rows, details = [], []
    # Include -7/-8 and +7/+8 from the start: boundary coverage is cheap with SVD.
    coarse = np.power(10.0, np.arange(-8, 9, dtype=float))
    for degree in range(1, maximum_degree + 1):
        start = time.perf_counter()
        caches = []
        for fold, (training, validation) in enumerate(splits, 1):
            transform = preprocessing(degree)
            Z = transform.fit_transform(X[training])
            V = transform.transform(X[validation])
            mean_y = y[training].mean()
            U, singular, Vt = svd(Z, full_matrices=False, check_finite=False)
            projected_validation = V @ Vt.T
            projected_target = U.T @ (y[training] - mean_y)
            caches.append((fold, validation, singular, projected_validation,
                           projected_target, mean_y))
        evaluated = set()

        def evaluate(alphas):
            for alpha in sorted(set(float(a) for a in alphas) - evaluated):
                for fold, validation, singular, pv, py, mean_y in caches:
                    prediction = pv @ (singular / (singular**2 + alpha) * py) + mean_y
                    rows.append(metric_row(degree, 'Ridge', alpha, 0, fold,
                                           y[validation], prediction))
                evaluated.add(alpha)

        evaluate(coarse)
        best = rank(summarize([r for r in rows if r['degree'] == degree])).iloc[0]
        center = np.log10(best.alpha)
        evaluate(10.0 ** np.clip(np.arange(center - 0.75, center + 1, 0.25), -8, 8))
        best = rank(summarize([r for r in rows if r['degree'] == degree])).iloc[0]
        elapsed = time.perf_counter() - start
        details.append(dict(degree=degree, features=int(Z.shape[1]), seconds=elapsed))
        print(f'  Ridge degree {degree:2}: CV MSE {best.mean_mse:.6g}, '
              f'alpha {best.alpha:.4g}, {elapsed:.1f}s', flush=True)
    return rows, details


def sparse_search(X, y, splits, degrees, settings):
    rows = []
    # A fixed moderate budget prevents ill-conditioned candidates consuming days.
    # Failed fits remain in the logs but cannot win, even if their MSE is small.
    alphas = 10.0 ** np.arange(6, -9, -1, dtype=float)
    for degree in degrees:
        start = time.perf_counter()
        for fold, (training, validation) in enumerate(splits, 1):
            transform = preprocessing(degree)
            Z = np.asfortranarray(transform.fit_transform(X[training]))
            V = transform.transform(X[validation])
            for method, ratio in [('LASSO', 1.0)] + [('ElasticNet', r) for r in settings.ratios]:
                model = make_model(method, float(alphas[0]), ratio, settings)
                for alpha in alphas:
                    model.set_params(alpha=float(alpha))
                    tick = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always', ConvergenceWarning)
                        model.fit(Z, y[training])
                    ok = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                    prediction = model.predict(V)
                    ok = ok and np.isfinite(prediction).all() and model.n_iter_ < settings.max_iter
                    rows.append(metric_row(degree, method, alpha, ratio, fold,
                                           y[validation], prediction, ok, model.n_iter_,
                                           model.dual_gap_, time.perf_counter() - tick))
            print(f'  LASSO/Elastic Net degree {degree}, fold {fold}/{len(splits)} '
                  f'finished ({time.perf_counter() - start:.1f}s)', flush=True)
    return rows


def fit_checked(pipeline, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        pipeline.fit(X, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    model = pipeline.named_steps['regression']
    if isinstance(model, (Lasso, ElasticNet)):
        converged = converged and model.n_iter_ < model.max_iter
    if not converged:
        raise RuntimeError('Selected model did not converge on refit. No submission was written.')
    return pipeline
