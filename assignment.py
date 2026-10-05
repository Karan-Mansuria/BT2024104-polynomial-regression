"""Train independent polynomial models and write verified y-only submissions."""
import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from search import (Settings, fit_checked, make_pipeline, rank, ridge_search,
                    sparse_search, summarize)

ROLL = 'BT2024104'
PROBLEMS = {'var1': (6, 10), 'var2': (3, 20)}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load_numeric(path, columns):
    frame = pd.read_csv(path)
    if frame.columns.tolist() != columns:
        raise ValueError(f'{path}: expected columns {columns}, found {frame.columns.tolist()}')
    frame = frame.apply(pd.to_numeric, errors='raise')
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f'{path}: missing or nonfinite data')
    return frame


def predict(problem, data, output):
    directory = output / problem
    info = json.loads((directory / 'selection.json').read_text())
    model = joblib.load(directory / 'model.joblib')
    frame = load_numeric(data / f'{ROLL}_test_{problem}.csv', info['features'])
    template = pd.read_csv(data / 'sample_submission.csv')
    if template.columns.tolist() != ['y'] or len(template) != len(frame):
        raise ValueError('Sample submission must have one y column and match the test row count.')
    values = model.predict(frame.to_numpy(dtype=float))
    if not np.isfinite(values).all():
        raise ValueError('Predictions must all be finite.')
    destination = output / f'{ROLL}_pred_{problem}.csv'
    pd.DataFrame({'y': values}).to_csv(destination, index=False)
    reloaded = pd.read_csv(destination, float_precision='round_trip')
    if reloaded.columns.tolist() != ['y'] or len(reloaded) != len(frame):
        raise AssertionError('Prediction shape verification failed.')
    np.testing.assert_allclose(reloaded.y, values, rtol=1e-12, atol=1e-12)
    # Independently check predictions in separate contiguous batches: row order
    # must agree with both the saved pipeline and the untouched test file.
    batched = np.concatenate([model.predict(part) for part in
                              np.array_split(frame.to_numpy(dtype=float), 7)])
    np.testing.assert_allclose(reloaded.y, batched, rtol=1e-10, atol=1e-10)
    verification = dict(rows=len(frame), columns=['y'], finite=True,
                        matches_saved_model=True, test_order_preserved=True,
                        sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    save_json(directory / 'prediction_verification.json', verification)
    print(f'  Verified {destination}: {len(frame)} predictions', flush=True)


def train(problem, data, output, settings):
    start = time.perf_counter()
    directory = output / problem
    directory.mkdir(parents=True, exist_ok=True)
    count, maximum_degree = PROBLEMS[problem]
    features = [f'x{i}' for i in range(1, count + 1)]
    source = data / f'{ROLL}_train_{problem}.csv'
    frame = load_numeric(source, features + ['y'])
    X, y = frame[features].to_numpy(dtype=float), frame.y.to_numpy(dtype=float)
    splits = list(KFold(settings.folds, shuffle=True, random_state=settings.seed).split(X))
    fold_ids = np.empty(len(y), dtype=int)
    for fold, (_, validation) in enumerate(splits, 1):
        fold_ids[validation] = fold
    pd.DataFrame({'row': np.arange(len(y)), 'validation_fold': fold_ids}).to_csv(
        directory / 'cv_splits.csv', index=False)
    save_json(directory / 'audit.json', dict(rows=len(frame), features=features,
        duplicate_rows=int(frame.duplicated().sum()),
        duplicate_inputs=int(frame.duplicated(features).sum()),
        training_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        settings=asdict(settings)))
    frame.describe().to_csv(directory / 'training_summary.csv')
    print(f'{problem}: {len(y)} training rows, {count} inputs; degrees 1-{maximum_degree}', flush=True)
    ridge_rows, timings = ridge_search(X, y, splits, maximum_degree)
    ridge_results = summarize(ridge_rows)
    degrees = rank(ridge_results).drop_duplicates('degree').head(settings.shortlist).degree.tolist()
    pd.DataFrame(ridge_rows).to_csv(directory / 'ridge_fold_results.csv', index=False)
    ridge_results.to_csv(directory / 'ridge_cv_results.csv', index=False)
    pd.DataFrame(timings).to_csv(directory / 'ridge_runtime.csv', index=False)
    print(f'  Compare LASSO and Elastic Net at shortlisted degrees {degrees}', flush=True)
    rows = ridge_rows + sparse_search(X, y, splits, degrees, settings)
    results = summarize(rows)
    pd.DataFrame(rows).to_csv(directory / 'fold_results.csv', index=False)
    results.to_csv(directory / 'cv_results.csv', index=False)
    selected = rank(results).iloc[0]
    specification = dict(degree=int(selected.degree), method=str(selected.method),
                         alpha=float(selected.alpha), ratio=float(selected.l1_ratio))
    # These OOF scores are selection-CV diagnostics, not an unbiased nested-CV
    # estimate: the configuration was chosen using these same validation folds.
    oof = np.empty_like(y)
    selected_folds = []
    for fold, (training, validation) in enumerate(splits, 1):
        model = fit_checked(make_pipeline(**specification, settings=settings), X[training], y[training])
        oof[validation] = model.predict(X[validation])
        selected_folds.append(dict(fold=fold, mse=mean_squared_error(y[validation], oof[validation]),
                                   r2=r2_score(y[validation], oof[validation])))
    pd.DataFrame(selected_folds).to_csv(directory / 'selected_fold_metrics.csv', index=False)
    pd.DataFrame({'row': np.arange(len(y)), 'fold': fold_ids, 'y_true': y,
                  'y_pred': oof}).to_csv(directory / 'oof_predictions.csv', index=False)
    final = fit_checked(make_pipeline(**specification, settings=settings), X, y)
    joblib.dump(final, directory / 'model.joblib')
    info = specification | dict(problem=problem, features=features,
        polynomial_features=int(final.named_steps['polynomial'].n_output_features_),
        mean_cv_mse=float(selected.mean_mse), std_cv_mse=float(selected.std_mse),
        mean_cv_r2=float(selected.mean_r2), std_cv_r2=float(selected.std_r2),
        refit_mean_cv_mse=float(np.mean([r['mse'] for r in selected_folds])),
        oof_mse=float(mean_squared_error(y, oof)), oof_r2=float(r2_score(y, oof)),
        shortlist_degrees=[int(d) for d in degrees],
        failed_sparse_fold_fits=int(sum(not r['converged'] for r in rows)),
        boundary_selected=bool(np.isclose(np.log10(selected.alpha), -8) or
                               np.isclose(np.log10(selected.alpha), 8)),
        final_fit_converged=True,
        cv_interpretation='Selection CV; not an independent or nested estimate.',
        training_seconds=float(time.perf_counter() - start))
    save_json(directory / 'selection.json', info)
    from reporting import plots
    plots(results, y, oof, directory, problem)
    # Test features are first opened here, after selection and final fitting.
    predict(problem, data, output)
    print(f'{problem} finished: {specification}, CV MSE {selected.mean_mse:.6g}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'predict'])
    parser.add_argument('--problem', choices=['both', 'var1', 'var2'], default='both')
    parser.add_argument('--data-dir', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        for problem in PROBLEMS if args.problem == 'both' else [args.problem]:
            if args.command == 'train':
                train(problem, args.data_dir, args.output, Settings())
            else:
                predict(problem, args.data_dir, args.output)


if __name__ == '__main__':
    main()
