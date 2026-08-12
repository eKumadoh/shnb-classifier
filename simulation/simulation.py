import time
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, Memory
from scipy.stats import multivariate_normal, wilcoxon
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from model.sp_hnb import SemiparametricHNB
from model.hnb import Hidden_NB
from model.kd_nb import KDENaiveBayes

np.random.seed(42)

_DIST_CODE = {"multimodal": 0}
_STRENGTH_CODE = {"weak": 0, "moderate": 1, "strong": 2}

STRENGTH_RANGES = {
    "weak": (0.10, 0.30),
    "moderate": (0.35, 0.60),
    "strong": (0.65, 0.85),
}

K_VALUES = (3, 5, 7, 10)
N_BINS_VALUES = (8, 10, 11, 13)

CACHE_DIR = "./shnb_sim_cache"
_memory = Memory(CACHE_DIR, verbose=0)


def make_rep_seed(n, p, dist, strength, rep):
    dist_code = _DIST_CODE[dist]
    strength_code = _STRENGTH_CODE[strength]
    seed = (((n * 31 + p) * 31 + dist_code) * 31 + strength_code) * 1000 + rep
    return seed % (2**32 - 1)


def generate_correlated_data(n, p, dist_type, strength, rep_seed):
    if strength not in STRENGTH_RANGES:
        raise ValueError(f"Unknown strength '{strength}', expected one of {list(STRENGTH_RANGES)}")
    if dist_type != "multimodal":
        raise ValueError(f"Unknown dist_type '{dist_type}', only 'multimodal' is supported")
    lo, hi = STRENGTH_RANGES[strength]

    loading_rng = np.random.RandomState(1000 + p * 10 + _STRENGTH_CODE[strength])
    loadings = loading_rng.uniform(lo, hi, size=p)

    Sigma = np.outer(loadings, loadings)
    np.fill_diagonal(Sigma, 1.0)

    data_rng = np.random.RandomState(rep_seed)

    X_list = []
    y_list = []

    for c in [0, 1]:
        mean = np.zeros(p) + c * 1.5

        mix = data_rng.binomial(1, 0.5, size=n // 2)
        X1 = multivariate_normal.rvs(mean=mean - 1.0, cov=Sigma, size=n // 2,
                                      random_state=data_rng)
        X2 = multivariate_normal.rvs(mean=mean + 1.0, cov=Sigma, size=n // 2,
                                      random_state=data_rng)
        Xc = mix[:, None] * X1 + (1 - mix)[:, None] * X2

        X_list.append(Xc)
        y_list.append(np.full(n // 2, c))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return X, y


def _confidence_and_correctness(y_true, y_proba):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.argmax(y_proba, axis=1)
    confidence = y_proba[np.arange(len(y_true)), y_pred]
    correct = (y_pred == y_true).astype(float)
    return confidence, correct


def _expected_calibration_error(y_true, y_proba, n_bins=10):
    confidence, correct = _confidence_and_correctness(y_true, y_proba)
    n = len(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & (confidence <= hi if i == n_bins - 1 else confidence < hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        ece += (n_in_bin / n) * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece)


def _maximum_calibration_error(y_true, y_proba, n_bins=10):
    confidence, correct = _confidence_and_correctness(y_true, y_proba)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    gaps = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & (confidence <= hi if i == n_bins - 1 else confidence < hi)
        if mask.sum() == 0:
            continue
        gaps.append(abs(confidence[mask].mean() - correct[mask].mean()))
    return float(max(gaps)) if gaps else 0.0


_ALL_METRIC_KEYS = ['log_loss', 'ece', 'mce']


def compute_metrics(y_true, y_proba):
    """Log-loss, Expected Calibration Error, and Maximum Calibration Error."""
    metrics = {}

    try:
        if y_proba is not None:
            metrics['log_loss'] = log_loss(y_true, y_proba)
            metrics['ece'] = _expected_calibration_error(y_true, y_proba)
            metrics['mce'] = _maximum_calibration_error(y_true, y_proba)
        else:
            for key in _ALL_METRIC_KEYS:
                metrics[key] = np.nan

    except Exception as e:
        print(f"    [WARNING] Metric computation failed: {e}")
        for key in _ALL_METRIC_KEYS:
            metrics[key] = np.nan

    return metrics


def _fit_predict_timed(model, X_train, y_train, X_test):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    predict_time = time.perf_counter() - t0

    return y_pred, y_proba, fit_time, predict_time


def _fit_one(name, model, X_train, y_train, X_test, y_test, result_row):
    try:
        y_pred, y_proba, fit_time, predict_time = _fit_predict_timed(
            model, X_train, y_train, X_test
        )
        metrics = compute_metrics(y_test, y_proba)
        for metric, value in metrics.items():
            result_row[f'{name}_{metric}'] = value
        result_row[f'{name}_y_pred'] = np.asarray(y_pred).tolist()
        result_row[f'{name}_fit_time'] = fit_time
        result_row[f'{name}_predict_time'] = predict_time

    except Exception as e:
        print(f"    [ERROR] {name}: {e}")
        for metric in _ALL_METRIC_KEYS:
            result_row[f'{name}_{metric}'] = np.nan
        result_row[f'{name}_y_pred'] = np.nan
        result_row[f'{name}_fit_time'] = np.nan
        result_row[f'{name}_predict_time'] = np.nan


@_memory.cache
def _run_one_replication(n, p, dist, strength, rep, k_values, n_bins_values):
    rep_seed = make_rep_seed(n, p, dist, strength, rep)
    X, y = generate_correlated_data(n, p, dist, strength, rep_seed=rep_seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=rep
    )

    result_row = {"n": n, "p": p, "dist": dist, "strength": strength, "rep": rep}

    for k in k_values:
        model = SemiparametricHNB(k_neighbors=k, ucv_grid_points=20)
        _fit_one(f'SHNB_k{k}', model, X_train, y_train, X_test, y_test, result_row)

    for bins in n_bins_values:
        model = Hidden_NB(n_bins=bins)
        _fit_one(f'HNB_bins{bins}', model, X_train, y_train, X_test, y_test, result_row)

    _fit_one('GNB', GaussianNB(), X_train, y_train, X_test, y_test, result_row)
    _fit_one('KDE_NB', KDENaiveBayes(), X_train, y_train, X_test, y_test, result_row)

    return result_row


def run_simulation(k_values=K_VALUES, n_bins_values=N_BINS_VALUES, n_jobs=-1, verbose=10):
    sample_sizes = [100, 500, 1000, 5000]
    feature_sizes = [2, 5, 10]
    dists = ["multimodal"]
    strengths = ["weak", "moderate", "strong"]
    n_replications = 50

    tasks = list(itertools.product(
        sample_sizes, feature_sizes, dists, strengths, range(n_replications)
    ))
    print(f"Running {len(tasks)} replications x "
          f"({len(k_values)} SHNB k-values + {len(n_bins_values)} HNB bin-counts + 2) "
          f"= {len(tasks) * (len(k_values) + len(n_bins_values) + 2)} total model fits, "
          f"across n_jobs={n_jobs} workers (cache: {CACHE_DIR})...")

    start_time = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_run_one_replication)(n, p, dist, strength, rep, k_values, n_bins_values)
        for n, p, dist, strength, rep in tasks
    )

    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return pd.DataFrame(results)


def summarize_results(df_results):
    metric_cols = [col for col in df_results.columns
                   if any(model in col for model in ['SHNB_k', 'GNB', 'KDE_NB', 'HNB_bins'])]

    agg_dict = {col: ['mean', 'std', 'count'] for col in metric_cols}
    summary = df_results.groupby(['n', 'p', 'dist', 'strength']).agg(agg_dict).reset_index()
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                       for col in summary.columns.values]

    for col in metric_cols:
        mean_col, std_col, count_col = f'{col}_mean', f'{col}_std', f'{col}_count'
        if all(c in summary.columns for c in [mean_col, std_col, count_col]):
            summary[f'{col}_mcse'] = summary[std_col] / np.sqrt(summary[count_col])

    return summary


def compute_significance(df_results, best_shnb_col='SHNB_k5_log_loss',
                          baseline_cols=('HNB_bins10_log_loss', 'GNB_log_loss', 'KDE_NB_log_loss'),
                          alpha=0.05):
    rows = []
    for (n, p, dist, strength), group in df_results.groupby(['n', 'p', 'dist', 'strength']):
        row = {"n": n, "p": p, "dist": dist, "strength": strength}
        for base_col in baseline_cols:
            paired = group[[best_shnb_col, base_col]].dropna()
            if len(paired) < 2:
                row[f'{best_shnb_col}_lt_{base_col}_pvalue'] = np.nan
                row[f'{best_shnb_col}_lt_{base_col}_significant'] = False
                continue
            try:
                stat, pvalue = wilcoxon(paired[best_shnb_col], paired[base_col],
                                         alternative='less')
            except ValueError:
                pvalue = np.nan
            row[f'{best_shnb_col}_lt_{base_col}_pvalue'] = pvalue
            row[f'{best_shnb_col}_lt_{base_col}_significant'] = bool(pvalue is not np.nan and pvalue < alpha)
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_results = run_simulation()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    detailed_file = f"simulation_results_detailed_{timestamp}.csv"
    df_results.to_csv(detailed_file, index=False)
    print(f"\nDetailed results saved to '{detailed_file}'")

    df_summary = summarize_results(df_results)
    summary_file = f"simulation_results_summary_{timestamp}.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"Summary results (incl. MCSE) saved to '{summary_file}'")

    df_significance = compute_significance(df_results)
    significance_file = f"simulation_results_significance_{timestamp}.csv"
    df_significance.to_csv(significance_file, index=False)
    print(f"Significance results saved to '{significance_file}'")
