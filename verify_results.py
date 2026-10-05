"""Independent audit of completed search, selected models and submission files."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from threadpoolctl import threadpool_limits

from assignment import PROBLEMS, ROLL


def verify(data, output):
    report = {}
    for problem, (features, max_degree) in PROBLEMS.items():
        directory = output / problem
        selection = json.loads((directory / 'selection.json').read_text())
        audit = json.loads((directory / 'audit.json').read_text())
        train_path = data / f'{ROLL}_train_{problem}.csv'
        assert audit['training_sha256'] == hashlib.sha256(train_path.read_bytes()).hexdigest()
        train = pd.read_csv(train_path)
        test = pd.read_csv(data / f'{ROLL}_test_{problem}.csv')
        template = pd.read_csv(data / 'sample_submission.csv')
        predictions = pd.read_csv(output / f'{ROLL}_pred_{problem}.csv', float_precision='round_trip')
        assert predictions.columns.tolist() == template.columns.tolist() == ['y']
        assert len(predictions) == len(template) == len(test)
        assert np.isfinite(predictions.y).all()
        columns = [f'x{i}' for i in range(1, features + 1)]
        assert test.columns.tolist() == selection['features'] == columns
        model = joblib.load(directory / 'model.joblib')
        np.testing.assert_allclose(predictions.y, model.predict(test.to_numpy()), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(model.named_steps['raw_scaler'].mean_, train[columns].mean(),
                                   rtol=1e-12, atol=1e-12)
        assert model.named_steps['polynomial'].degree == selection['degree']
        results = pd.read_csv(directory / 'cv_results.csv', float_precision='round_trip')
        folds = pd.read_csv(directory / 'fold_results.csv', float_precision='round_trip')
        assert results.folds.eq(5).all()
        assert set(results.loc[results.method == 'Ridge', 'degree']) == set(range(1, max_degree + 1))
        for method in ['LASSO', 'ElasticNet']:
            assert set(results.loc[results.method == method, 'degree']) == set(selection['shortlist_degrees'])
        eligible = results.loc[results.eligible]
        np.testing.assert_allclose(selection['mean_cv_mse'], eligible.mean_mse.min(), rtol=1e-12)
        chosen = results.loc[(results.degree == selection['degree']) &
                            (results.method == selection['method']) &
                            np.isclose(results.alpha, selection['alpha'], rtol=1e-12, atol=0) &
                            np.isclose(results.l1_ratio, selection['ratio'])]
        assert len(chosen) == 1 and bool(chosen.iloc[0].eligible)
        keys = ['degree', 'method', 'alpha', 'l1_ratio']
        recomputed = folds.groupby(keys).agg(mean_mse=('mse', 'mean'),
                                            converged_folds=('converged', 'sum'))
        recorded = results.set_index(keys).sort_index()
        np.testing.assert_allclose(recorded.mean_mse, recomputed.mean_mse, rtol=1e-12)
        np.testing.assert_array_equal(recorded.converged_folds, recomputed.converged_folds)
        splits = pd.read_csv(directory / 'cv_splits.csv').sort_values('row')
        oof = pd.read_csv(directory / 'oof_predictions.csv').sort_values('row')
        np.testing.assert_array_equal(oof.row, np.arange(len(train)))
        np.testing.assert_array_equal(oof.fold, splits.validation_fold)
        np.testing.assert_allclose(oof.y_true, train.y)
        assert set(oof.fold) == set(range(1, 6))
        np.testing.assert_allclose(mean_squared_error(oof.y_true, oof.y_pred), selection['oof_mse'])
        report[problem] = dict(verified=True, prediction_rows=len(predictions),
                               candidates=len(results), fold_evaluations=len(folds),
                               selected_method=selection['method'], selected_degree=selection['degree'])
    (output / 'verification_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        verify(args.data_dir, args.output)
