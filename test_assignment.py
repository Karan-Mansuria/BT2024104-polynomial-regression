"""Small correctness tests; no expensive assignment-wide training."""
import unittest

import numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from search import Settings, make_pipeline, preprocessing, rank, ridge_search, summarize


class RegressionTests(unittest.TestCase):
    def test_svd_grid_matches_independent_sklearn_ridge(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(45, 3))
        y = X[:, 0]**2 - 2 * X[:, 1] + rng.normal(scale=.1, size=45)
        splits = list(KFold(5, shuffle=True, random_state=4).split(X))
        with threadpool_limits(limits=1):
            rows, _ = ridge_search(X, y, splits, 3)
            for degree in (1, 3):  # Both tall and wide polynomial designs.
                for alpha in (1e-6, 1.0, 1e6):
                    for fold, (tr, va) in enumerate(splits, 1):
                        model = make_pipeline(degree, 'Ridge', alpha, 0, Settings()).fit(X[tr], y[tr])
                        expected = np.mean((y[va] - model.predict(X[va]))**2)
                        actual = next(r['mse'] for r in rows if r['degree'] == degree
                                      and r['alpha'] == alpha and r['fold'] == fold)
                        self.assertAlmostEqual(actual, expected, places=7)

    def test_validation_cannot_change_scaler_statistics(self):
        X = np.arange(30.0).reshape(15, 2)
        transform = preprocessing(3).fit(X[:10])
        raw_mean = transform.named_steps['raw_scaler'].mean_.copy()
        polynomial_mean = transform.named_steps['polynomial_scaler'].mean_.copy()
        transform.transform(X[10:] + 10000)
        np.testing.assert_allclose(raw_mean, X[:10].mean(axis=0))
        np.testing.assert_array_equal(raw_mean, transform.named_steps['raw_scaler'].mean_)
        np.testing.assert_array_equal(polynomial_mean, transform.named_steps['polynomial_scaler'].mean_)

    def test_nonconverged_candidate_cannot_win(self):
        rows = []
        for alpha, mse, ok in [(1.0, 5.0, True), (.01, .01, False)]:
            for fold in range(1, 6):
                rows.append(dict(degree=2, method='LASSO', alpha=alpha, l1_ratio=1.0,
                    fold=fold, mse=mse, r2=.5, converged=ok or fold != 3,
                    seconds=0, iterations=1))
        self.assertEqual(float(rank(summarize(rows)).iloc[0].alpha), 1.0)


if __name__ == '__main__':
    unittest.main()
