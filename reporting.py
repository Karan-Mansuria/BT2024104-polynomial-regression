"""Two compact figures per problem for the assignment report."""
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'bt2024104-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plots(results, y, oof, directory, problem):
    eligible = results.loc[results.eligible]
    best = eligible.sort_values('mean_mse').drop_duplicates(['degree', 'method'])
    best.sort_values(['method', 'degree']).to_csv(directory / 'degree_method_summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), layout='constrained')
    shortlist = sorted(best.loc[best.method != 'Ridge', 'degree'].unique())
    styles = {
        'Ridge': dict(color='tab:green', marker='s', markersize=4, linestyle='-', zorder=2),
        'LASSO': dict(color='tab:orange', marker='x', markersize=6, linestyle=':', zorder=3),
        'ElasticNet': dict(color='tab:blue', marker='o', markersize=9, linestyle='--',
                           markerfacecolor='none', markeredgewidth=1.5, zorder=4),
    }
    for method in styles:
        group = best.loc[best.method == method].sort_values('degree')
        label = 'Elastic Net' if method == 'ElasticNet' else method
        axes[0].plot(group.degree, group.mean_mse, label=label, **styles[method])
        close = group.loc[group.degree.isin(shortlist)]
        axes[1].plot(close.degree, close.mean_mse, label=label, **styles[method])
    axes[0].set(title='All degrees', ylabel='Five-fold selection CV MSE', yscale='log')
    axes[1].set(title='Shortlisted degrees (zoom)', xticks=shortlist)
    # Zoom to the sparse-model scores: a distant Ridge score may be outside
    # this panel, but remains visible on the complete degree curve at left.
    sparse = best.loc[best.method != 'Ridge', 'mean_mse']
    span = max(float(sparse.max() - sparse.min()), float(sparse.mean()) * .005)
    axes[1].set_ylim(float(sparse.min()) - span * .25, float(sparse.max()) + span * .25)
    for ax in axes:
        ax.set_xlabel('Polynomial degree')
        ax.grid(alpha=.2)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{problem}: polynomial degree comparison', fontsize=11)
    fig.savefig(directory / 'degree_comparison.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), layout='constrained')
    axes[0].scatter(y, oof, s=8, alpha=.5)
    limits = [min(y.min(), oof.min()), max(y.max(), oof.max())]
    axes[0].plot(limits, limits, '--', color='black', linewidth=1)
    axes[0].set(xlabel='Observed y', ylabel='Out-of-fold prediction')
    axes[1].scatter(oof, y-oof, s=8, alpha=.5)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set(xlabel='Out-of-fold prediction', ylabel='Residual')
    fig.suptitle(f'{problem}: selection-CV diagnostics')
    fig.savefig(directory / 'oof_diagnostics.png', dpi=180)
    plt.close(fig)
