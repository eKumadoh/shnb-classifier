import time
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, Memory
from scipy.stats import skew, kurtosis, norm, spearmanr
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

from model.sp_hnb import SemiparametricHNB
from model.hnb import Hidden_NB
from model.hnb_EF import Hidden_NB_EqualFrequency
from model.mdlp import Hidden_NB_Supervised
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
N_BINS_VALUES = (8, 10, 11)

TEST_SIZE = 2000
_TEST_SEED_OFFSET = 10_000_000  # keeps the held-out test draw independent of the training draw

# Number of simulation replications actually used to produce a run's CSVs.
# (The manuscript's methods text and appendix table captions disagree with
# each other on this number -- 50 vs 100 -- so this constant is the single
# source of truth; it is printed at the start of run_simulation().)
N_REPLICATIONS = 50

CACHE_DIR = "./shnb_sim_cache"
_memory = Memory(CACHE_DIR, verbose=0)


def make_rep_seed(n, p, dist, strength, rep):
    dist_code = _DIST_CODE[dist]
    strength_code = _STRENGTH_CODE[strength]
    seed = (((n * 31 + p) * 31 + dist_code) * 31 + strength_code) * 1000 + rep
    return seed % (2**32 - 1)


# Fixed bimodal target marginal (Algorithm dgp, Step 3): equal-weight
# mixture of N(-2.5, 1) and N(2.5, 1), bimodality coefficient ~0.58, fixed
# across every scenario regardless of n, p, strength.
_BIMODAL_MEANS = (-2.5, 2.5)
_BIMODAL_SD = 1.0


def _bimodal_cdf(x):
    """CDF of the fixed bimodal target marginal (Algorithm dgp, Step 3)."""
    m1, m2 = _BIMODAL_MEANS
    return 0.5 * norm.cdf(x, loc=m1, scale=_BIMODAL_SD) + 0.5 * norm.cdf(x, loc=m2, scale=_BIMODAL_SD)


def _bimodal_logpdf(x):
    """Log-density of the fixed bimodal target marginal; used by the oracle
    Bayes classifier below."""
    m1, m2 = _BIMODAL_MEANS
    lp1 = norm.logpdf(x, loc=m1, scale=_BIMODAL_SD)
    lp2 = norm.logpdf(x, loc=m2, scale=_BIMODAL_SD)
    stacked = np.stack([lp1, lp2], axis=-1) + np.log(0.5)
    return logsumexp(stacked, axis=-1)


def _bimodal_ppf(u, lo=-15.0, hi=15.0, n_iter=60):
    """Inverse CDF of the fixed bimodal target marginal via vectorised
    bisection against _bimodal_cdf (the mixture CDF has no closed-form
    inverse). Density is negligible outside +/-15, so bisection on that
    interval converges reliably; 60 iterations gives ~1e-18 precision."""
    u = np.clip(u, 1e-15, 1 - 1e-15)
    lo_arr = np.full_like(u, lo, dtype=float)
    hi_arr = np.full_like(u, hi, dtype=float)
    for _ in range(n_iter):
        mid = 0.5 * (lo_arr + hi_arr)
        cdf_mid = _bimodal_cdf(mid)
        go_right = cdf_mid < u
        lo_arr = np.where(go_right, mid, lo_arr)
        hi_arr = np.where(go_right, hi_arr, mid)
    return 0.5 * (lo_arr + hi_arr)


def generate_correlated_data(n, p, dist_type, strength, rep_seed):
    """Implements Algorithm dgp (paper Algorithm 1):

      Step 1 -- Latent dependence structure: loadings -> R (outer product,
                diagonal reset to 1) -> Cholesky factor L. Fixed per
                (p, strength), shared across all n and replications.
      Step 2 -- Correlated uniforms (Gaussian copula): per class, draw iid
                standard normals, correlate via L, map through Phi.
      Step 3 -- Bimodal marginal transform: apply the inverse CDF of the
                fixed bimodal target marginal, elementwise and
                independently per feature.
      Step 4 -- Class separation: shift every feature of class 1 by the
                constant +1.5.

    Returns (X, y, R): R is the latent Gaussian correlation matrix used to
    build L (what the paper calls R in Algorithm dgp), kept under the name
    `Sigma` at call sites for signature compatibility. It is NOT the
    Pearson correlation of the returned bimodal features (see Section 2.4,
    Step 3 discussion)."""
    if strength not in STRENGTH_RANGES:
        raise ValueError(f"Unknown strength '{strength}', expected one of {list(STRENGTH_RANGES)}")
    if dist_type != "multimodal":
        raise ValueError(f"Unknown dist_type '{dist_type}', only 'multimodal' is supported")
    lo, hi = STRENGTH_RANGES[strength]

    # Step 1: loadings -> R -> L.
    loading_rng = np.random.RandomState(1000 + p * 10 + _STRENGTH_CODE[strength])
    loadings = loading_rng.uniform(lo, hi, size=p)
    R = np.outer(loadings, loadings)
    np.fill_diagonal(R, 1.0)
    L = np.linalg.cholesky(R)

    data_rng = np.random.RandomState(rep_seed)

    X_list = []
    y_list = []

    for c in [0, 1]:
        n_c = n // 2

        # Step 2: iid standard normals -> correlated via L -> uniforms via Phi.
        Z = data_rng.standard_normal(size=(n_c, p))
        Z_tilde = Z @ L.T  # row i is L @ z_i; Corr(Z_tilde) = R
        U = norm.cdf(Z_tilde)

        # Step 3: elementwise inverse-CDF of the fixed bimodal marginal.
        X0 = _bimodal_ppf(U)

        # Step 4: additive class shift (class 0 unshifted).
        Xc = X0 + c * 1.5

        X_list.append(Xc)
        y_list.append(np.full(n_c, c))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return X, y, R


def _scenario_diagnostics(X, p):
    """Realized Pearson correlation, realized Spearman rank correlation, and
    realized bimodality coefficient, computed on the pooled train+test data.

    Spearman is the quantity Algorithm dgp's construction is claimed to
    preserve (in expectation) relative to R; Pearson is reported only as an
    empirical diagnostic and is not expected to equal R (see Step 3
    discussion in generate_correlated_data)."""
    n = X.shape[0]
    diag = {}

    if p > 1:
        corr = np.corrcoef(X, rowvar=False)
        iu = np.triu_indices(p, k=1)
        diag['realized_corr_mean'] = float(corr[iu].mean())
        diag['realized_corr_max'] = float(corr[iu].max())

        spearman_res = spearmanr(X)
        spearman_corr = spearman_res.correlation if hasattr(spearman_res, 'correlation') else spearman_res[0]
        if p == 2:
            # scipy returns a scalar (not a matrix) when there are exactly
            # two variables.
            spearman_mat = np.array([[1.0, spearman_corr], [spearman_corr, 1.0]])
        else:
            spearman_mat = np.asarray(spearman_corr)
        diag['realized_spearman_mean'] = float(spearman_mat[iu].mean())
        diag['realized_spearman_max'] = float(spearman_mat[iu].max())
    else:
        diag['realized_corr_mean'] = np.nan
        diag['realized_corr_max'] = np.nan
        diag['realized_spearman_mean'] = np.nan
        diag['realized_spearman_max'] = np.nan

    bcs = []
    if n > 3:
        for j in range(p):
            s = skew(X[:, j])
            # Sarle's bimodality coefficient is
            #   BC = (g1^2 + 1) / (g2 + 3(n-1)^2/((n-2)(n-3)))
            # where g2 is EXCESS kurtosis, not the raw (non-excess) kurtosis
            # a previous version used here -- that added a spurious +3 to
            # the denominator and understated BC for every scenario.
            # fisher=True returns excess kurtosis directly.
            k_excess = kurtosis(X[:, j], fisher=True)
            denom = k_excess + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
            bcs.append((s ** 2 + 1) / denom if denom > 0 else np.nan)
    diag['bimodality_coef_mean'] = float(np.nanmean(bcs)) if bcs else np.nan

    return diag


def _confidence_and_correctness(y_true, y_proba):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.argmax(y_proba, axis=1)
    confidence = y_proba[np.arange(len(y_true)), y_pred]
    correct = (y_pred == y_true).astype(float)
    return confidence, correct


def _bin_edges(n_bins=10):
    return np.linspace(0.0, 1.0, n_bins + 1)


def _bin_mask(confidence, edges, i, n_bins):
    """Shared bin-membership definition used by ECE, MCE, and the
    reliability-diagram diagnostic below, so they can't silently diverge on
    binning convention: interior edges are assigned to the upper bin;
    confidence == 1.0 is assigned to the last bin."""
    lo, hi = edges[i], edges[i + 1]
    if i == n_bins - 1:
        return (confidence >= lo) & (confidence <= hi)
    return (confidence >= lo) & (confidence < hi)


def _expected_calibration_error(y_true, y_proba, n_bins=10):
    confidence, correct = _confidence_and_correctness(y_true, y_proba)
    n = len(y_true)
    edges = _bin_edges(n_bins)
    ece = 0.0
    for i in range(n_bins):
        mask = _bin_mask(confidence, edges, i, n_bins)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        ece += (n_in_bin / n) * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece)


def _maximum_calibration_error(y_true, y_proba, n_bins=10):
    confidence, correct = _confidence_and_correctness(y_true, y_proba)
    edges = _bin_edges(n_bins)
    gaps = []
    for i in range(n_bins):
        mask = _bin_mask(confidence, edges, i, n_bins)
        if mask.sum() == 0:
            continue
        gaps.append(abs(confidence[mask].mean() - correct[mask].mean()))
    return float(max(gaps)) if gaps else 0.0


def _reliability_diagram_data(y_true, y_proba, n_bins=10):
    """Per-bin confidence, empirical accuracy, occupancy count, and |gap|,
    using the same binning as ECE/MCE. Supports the "where miscalibration
    concentrates" claims and shows how many test points land in each bin
    (MCE is unstable when that count is small)."""
    confidence, correct = _confidence_and_correctness(y_true, y_proba)
    edges = _bin_edges(n_bins)
    rows = []
    for i in range(n_bins):
        mask = _bin_mask(confidence, edges, i, n_bins)
        n_in_bin = int(mask.sum())
        mean_conf = float(confidence[mask].mean()) if n_in_bin else np.nan
        acc = float(correct[mask].mean()) if n_in_bin else np.nan
        rows.append({
            "bin_index": i,
            "bin_lo": float(edges[i]),
            "bin_hi": float(edges[i + 1]),
            "n_in_bin": n_in_bin,
            "mean_confidence": mean_conf,
            "empirical_accuracy": acc,
            "abs_gap": float(abs(mean_conf - acc)) if n_in_bin else np.nan,
        })
    return pd.DataFrame(rows)


_ALL_METRIC_KEYS = ['log_loss', 'ece', 'mce', 'accuracy', 'auc']


def compute_metrics(y_true, y_proba):
    """Log-loss, ECE, MCE (primary calibration measures), plus accuracy and
    AUC (secondary discrimination measures)."""
    metrics = {}

    try:
        if y_proba is not None:
            y_pred = np.argmax(y_proba, axis=1)
            metrics['log_loss'] = log_loss(y_true, y_proba)
            metrics['ece'] = _expected_calibration_error(y_true, y_proba)
            metrics['mce'] = _maximum_calibration_error(y_true, y_proba)
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            for key in _ALL_METRIC_KEYS:
                metrics[key] = np.nan

    except Exception as e:
        print(f"    [WARNING] Metric computation failed: {e}")
        for key in _ALL_METRIC_KEYS:
            metrics[key] = np.nan

    return metrics


class OracleBayesClassifier(BaseEstimator, ClassifierMixin):
    """Uses the TRUE data-generating process (Algorithm dgp) to evaluate the
    exact class-conditional density in closed form, via the standard
    Gaussian-copula density identity

        c(u; R) = |R|^{-1/2} exp(-0.5 * z^T (R^{-1} - I) z),   z = Phi^{-1}(u)
        f(x)    = c(F(x); R) * prod_i f_bimodal(x_i)

    and hence the true posterior P(y|x) via Bayes' rule with the known
    (equal) class priors. Valid as an oracle ONLY because the generator is
    known exactly in this simulation study -- no fitted model should ever
    beat it in expectation, and how close SHNB gets to it is itself
    informative.

    `R` must be the exact latent correlation matrix returned by
    generate_correlated_data() for the (p, strength) scenario being
    evaluated (the third return value, kept under the name `Sigma` at call
    sites)."""

    def __init__(self, R, class_shift=1.5, class_priors=(0.5, 0.5)):
        self.R = R
        self.class_shift = class_shift
        self.class_priors = class_priors

    def fit(self, X, y):
        # Nothing is estimated from the sample -- the oracle uses the known
        # generative parameters. X, y are accepted only so this class has
        # the same (fit, predict_proba) interface as every other model.
        R = np.asarray(self.R, dtype=float)
        p = R.shape[0]
        self.p_ = p
        self.R_inv_minus_I_ = np.linalg.inv(R) - np.eye(p)
        sign, logdet = np.linalg.slogdet(R)
        self.log_det_R_ = logdet
        self.classes_ = np.array([0, 1])
        return self

    def _class_conditional_logpdf(self, X, shift):
        X_shifted = X - shift
        U = _bimodal_cdf(X_shifted)
        U = np.clip(U, 1e-15, 1 - 1e-15)
        Z = norm.ppf(U)
        quad = np.einsum('ni,ij,nj->n', Z, self.R_inv_minus_I_, Z)
        log_copula = -0.5 * self.log_det_R_ - 0.5 * quad
        log_marginals = _bimodal_logpdf(X_shifted).sum(axis=1)
        return log_copula + log_marginals

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        log_p0 = np.log(self.class_priors[0]) + self._class_conditional_logpdf(X, 0.0)
        log_p1 = np.log(self.class_priors[1]) + self._class_conditional_logpdf(X, self.class_shift)
        m = np.maximum(log_p0, log_p1)
        e0 = np.exp(log_p0 - m)
        e1 = np.exp(log_p1 - m)
        total = e0 + e1
        return np.stack([e0 / total, e1 / total], axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class PriorOnlyClassifier(BaseEstimator, ClassifierMixin):
    """Predicts the empirical training class prior for every point,
    ignoring all features. The floor every fitted model should clear."""

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        self.class_priors_ = counts / counts.sum()
        return self

    def predict_proba(self, X):
        n = np.asarray(X).shape[0]
        return np.tile(self.class_priors_, (n, 1))

    def predict(self, X):
        n = np.asarray(X).shape[0]
        return np.full(n, self.classes_[np.argmax(self.class_priors_)])


def _fit_predict_timed(model, X_train, y_train, X_test):
    """Timing path for every model in the main comparison. Calls
    predict_proba() exactly once. Returns predict time both as the raw
    total-test-set elapsed time and as a per-observation figure, since both
    are used downstream under explicitly different, unambiguous names."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    n_test = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)

    t0 = time.perf_counter()
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    predict_time_total = time.perf_counter() - t0
    predict_time_per_obs = predict_time_total / max(n_test, 1)

    return y_proba, fit_time, predict_time_total, predict_time_per_obs


def _fit_one(name, model, X_train, y_train, X_test, y_test, result_row):
    try:
        y_proba, fit_time, predict_time_total, predict_time_per_obs = _fit_predict_timed(
            model, X_train, y_train, X_test
        )
        metrics = compute_metrics(y_test, y_proba)
        for metric, value in metrics.items():
            result_row[f'{name}_{metric}'] = value
        result_row[f'{name}_fit_time_seconds'] = fit_time
        result_row[f'{name}_predict_time_total_seconds'] = predict_time_total
        result_row[f'{name}_predict_time_per_obs_seconds'] = predict_time_per_obs

    except Exception as e:
        print(f"    [ERROR] {name}: {e}")
        for metric in _ALL_METRIC_KEYS:
            result_row[f'{name}_{metric}'] = np.nan
        result_row[f'{name}_fit_time_seconds'] = np.nan
        result_row[f'{name}_predict_time_total_seconds'] = np.nan
        result_row[f'{name}_predict_time_per_obs_seconds'] = np.nan


def _time_standalone_fit(name, model_ctor, X_train, y_train, X_test, result_row):
    """fit_family() amortizes/shares bandwidth-search cost across the SHNB(K)
    models, which is NOT what a user fitting only one K from scratch would
    pay. This times a fully independent, from-scratch fit/predict_proba
    call for the representative operating point. Metrics aren't recomputed
    here since the fit_family-based fit for the same K already produced
    identical metrics (only _compute_weights depends on K)."""
    try:
        model = model_ctor()
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        n_test = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
        t0 = time.perf_counter()
        model.predict_proba(X_test)
        predict_time_total = time.perf_counter() - t0

        result_row[f'{name}_fit_time_seconds'] = fit_time
        result_row[f'{name}_predict_time_total_seconds'] = predict_time_total
        result_row[f'{name}_predict_time_per_obs_seconds'] = predict_time_total / max(n_test, 1)
    except Exception as e:
        print(f"    [ERROR] {name}: {e}")
        result_row[f'{name}_fit_time_seconds'] = np.nan
        result_row[f'{name}_predict_time_total_seconds'] = np.nan
        result_row[f'{name}_predict_time_per_obs_seconds'] = np.nan


# Representative operating point used for the SHNB standalone-timing row.
_SHNB_TIMING_K = 5


@_memory.cache
def _run_one_replication(n, p, dist, strength, rep, k_values, n_bins_values):
    rep_seed = make_rep_seed(n, p, dist, strength, rep)

    X_train, y_train, R = generate_correlated_data(n, p, dist, strength, rep_seed=rep_seed)

    test_seed = (rep_seed + _TEST_SEED_OFFSET) % (2 ** 32 - 1)
    X_test, y_test, _ = generate_correlated_data(TEST_SIZE, p, dist, strength, rep_seed=test_seed)
    n_test = X_test.shape[0]

    result_row = {"n": n, "p": p, "dist": dist, "strength": strength, "rep": rep}

    diag = _scenario_diagnostics(np.vstack([X_train, X_test]), p)
    result_row.update(diag)

    # SHNB(K) family: fit_family() fits priors/Gaussian params/bandwidths/
    # precomputed densities once, then redoes only the K-dependent
    # _compute_weights() step per K.
    models_by_k, shared_fit_time = SemiparametricHNB.fit_family(
        X_train, y_train, k_values, ucv_grid_points=20
    )
    for k in k_values:
        name = f'SHNB_k{k}'
        model = models_by_k[k]
        try:
            t0 = time.perf_counter()
            y_proba = model.predict_proba(X_test)
            predict_time_total = time.perf_counter() - t0
            metrics = compute_metrics(y_test, y_proba)
            for metric, value in metrics.items():
                result_row[f'{name}_{metric}'] = value
            # Amortized/shared cost from fit_family, NOT what fitting only
            # this K from scratch would cost -- see the standalone K=5
            # timing fields below for the real standalone cost.
            result_row[f'{name}_fit_time_seconds_amortized'] = model.fit_time_seconds_
            result_row[f'{name}_predict_time_total_seconds'] = predict_time_total
            result_row[f'{name}_predict_time_per_obs_seconds'] = predict_time_total / max(n_test, 1)
        except Exception as e:
            print(f"    [ERROR] {name}: {e}")
            for metric in _ALL_METRIC_KEYS:
                result_row[f'{name}_{metric}'] = np.nan
            result_row[f'{name}_fit_time_seconds_amortized'] = np.nan
            result_row[f'{name}_predict_time_total_seconds'] = np.nan
            result_row[f'{name}_predict_time_per_obs_seconds'] = np.nan

    # Honest standalone-fit timing for the representative operating point
    # (K=5): a fully independent from-scratch fit, not sharing any
    # bandwidth-search cost with the other K values.
    _time_standalone_fit(
        f'SHNB_k{_SHNB_TIMING_K}_standalone',
        lambda: SemiparametricHNB(k_neighbors=_SHNB_TIMING_K, ucv_grid_points=20),
        X_train, y_train, X_test, result_row,
    )

    for bins in n_bins_values:
        _fit_one(f'HNB_bins{bins}', Hidden_NB(n_bins=bins),
                  X_train, y_train, X_test, y_test, result_row)
        _fit_one(f'HNBeqfreq_bins{bins}', Hidden_NB_EqualFrequency(n_bins=bins),
                  X_train, y_train, X_test, y_test, result_row)

    _fit_one('HNBsup', Hidden_NB_Supervised(), X_train, y_train, X_test, y_test, result_row)

    _fit_one('GNB', GaussianNB(), X_train, y_train, X_test, y_test, result_row)
    _fit_one('KDE_NB', KDENaiveBayes(), X_train, y_train, X_test, y_test, result_row)

    _fit_one('SNB', SemiparametricHNB(use_hidden_parents=False),
              X_train, y_train, X_test, y_test, result_row)

    # Floor (Prior) and ceiling (Oracle) baselines added to every results row.
    _fit_one('Prior', PriorOnlyClassifier(), X_train, y_train, X_test, y_test, result_row)
    oracle = OracleBayesClassifier(R=R, class_shift=1.5, class_priors=(0.5, 0.5))
    _fit_one('Oracle', oracle, X_train, y_train, X_test, y_test, result_row)

    return result_row


def run_simulation(k_values=K_VALUES, n_bins_values=N_BINS_VALUES, n_jobs=-1, verbose=10):
    sample_sizes = [100, 500, 1000]
    feature_sizes = [2, 5, 10]
    dists = ["multimodal"]
    strengths = ["weak", "moderate", "strong"]
    n_replications = N_REPLICATIONS

    tasks = list(itertools.product(
        sample_sizes, feature_sizes, dists, strengths, range(n_replications)
    ))
    # SHNB k-values (+1 standalone re-fit) + HNB variants (eq-width/eq-freq)
    # + HNBsup + GNB + KDE-NB + SNB ablation + Prior baseline + Oracle baseline.
    n_models_per_task = (
        len(k_values) + 1
        + 2 * len(n_bins_values)
        + 1  # HNBsup
        + 1  # GNB
        + 1  # KDE-NB
        + 1  # SNB ablation
        + 1  # Prior-only baseline
        + 1  # Oracle baseline
    )
    print(f"N_REPLICATIONS = {N_REPLICATIONS} (see module-level constant; this is the "
          f"number actually used to produce the CSVs from this run).")
    print(f"Running {len(tasks)} replications x "
          f"({len(k_values)} SHNB k-values [+1 standalone K={_SHNB_TIMING_K} timing re-fit] "
          f"+ {len(n_bins_values)} HNB bin-counts "
          f"+ {len(n_bins_values)} HNB-equal-frequency bin-counts + 1 HNB-supervised "
          f"+ 1 GNB + 1 KDE-NB + 1 SNB ablation + 1 Prior-only baseline + 1 Oracle baseline) "
          f"= {len(tasks) * n_models_per_task} total model fits, "
          f"across n_jobs={n_jobs} workers (cache: {CACHE_DIR})...")
    print(f"Test set size fixed at {TEST_SIZE} per scenario (independent of training n).")

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
                   if any(model in col for model in
                          ['SHNB_k', 'GNB', 'KDE_NB', 'HNB_bins', 'HNBeqfreq_bins', 'HNBsup', 'SNB',
                           'Prior', 'Oracle'])]

    agg_dict = {col: ['mean', 'std', 'count'] for col in metric_cols}
    summary = df_results.groupby(['n', 'p', 'dist', 'strength']).agg(agg_dict).reset_index()
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                       for col in summary.columns.values]

    for col in metric_cols:
        mean_col, std_col, count_col = f'{col}_mean', f'{col}_std', f'{col}_count'
        if all(c in summary.columns for c in [mean_col, std_col, count_col]):
            summary[f'{col}_mcse'] = summary[std_col] / np.sqrt(summary[count_col])

    return summary


def summarize_diagnostics(df_results):
    diag_cols = ['realized_corr_mean', 'realized_corr_max',
                 'realized_spearman_mean', 'realized_spearman_max',
                 'bimodality_coef_mean']
    return (
        df_results.groupby(['n', 'p', 'dist', 'strength'])[diag_cols]
        .agg(['mean', 'std'])
        .reset_index()
    )


def run_bivariate_bandwidth_sensitivity_check(n_values=(100, 500, 1000), n_reps=10):
    """Compares the production diagonal product-kernel bandwidth against a
    jointly-optimised bivariate UCV bandwidth, for p=2, across n in
    {100, 500, 1000} and all three dependence-strength levels, averaged
    over n_reps replications per (n, strength).

    Returns:
        df_raw:    one row per (n, strength, rep, class) -- the raw checks.
        per_cell:  mean/std/count of ucv_relative_gap per (n, strength).
        pooled:    a single dict {mean, std, n} pooling every (n, strength,
                   rep, class) draw.
    """
    rows = []
    for n in n_values:
        for strength in ["weak", "moderate", "strong"]:
            for rep in range(n_reps):
                rep_seed = make_rep_seed(n, 2, "multimodal", strength, rep)
                X_train, y_train, _ = generate_correlated_data(n, 2, "multimodal", strength, rep_seed)
                model = SemiparametricHNB(k_neighbors=5, ucv_grid_points=20).fit(X_train, y_train)
                check = model.bivariate_bandwidth_sensitivity_check(0, 1, grid_points=8)
                for c, res in check.items():
                    rows.append({"n": n, "strength": strength, "rep": rep, "class": c, **res})

    df_raw = pd.DataFrame(rows)

    per_cell = (
        df_raw.groupby(["n", "strength"])["ucv_relative_gap"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    pooled = {
        "pooled_mean_relative_gap": float(df_raw["ucv_relative_gap"].mean()),
        "pooled_std_relative_gap": float(df_raw["ucv_relative_gap"].std()),
        "pooled_n": int(df_raw["ucv_relative_gap"].count()),
    }

    return df_raw, per_cell, pooled


def run_reliability_diagnostics(models_to_check=("SHNB_k5", "HNB_bins10"), n_reps=20, n_bins=10):
    """Reliability-diagram / bin-occupancy diagnostics for a small, named
    set of representative models, averaged over n_reps replications per
    scenario. Kept separate from the main _run_one_replication loop since
    storing full per-bin diagnostics for every model x every replication x
    all 27 scenarios would be enormous and isn't needed beyond supporting a
    specific localization claim about a couple of named models.

    Returns:
        df_raw:    one row per (n, p, strength, rep, model, bin) with
                   n_in_bin, mean_confidence, empirical_accuracy, abs_gap.
        df_pooled: the same, pooled over replications within each
                   (n, p, strength, model, bin), weighted by n_in_bin so
                   empty bins in a given replication don't distort the mean.
    """
    model_builders = {
        "SHNB_k5": lambda: SemiparametricHNB(k_neighbors=5, ucv_grid_points=20),
        "HNB_bins10": lambda: Hidden_NB(n_bins=10),
        "HNBeqfreq_bins10": lambda: Hidden_NB_EqualFrequency(n_bins=10),
        "HNBsup": lambda: Hidden_NB_Supervised(),
        "GNB": lambda: GaussianNB(),
        "KDE_NB": lambda: KDENaiveBayes(),
        "SNB": lambda: SemiparametricHNB(use_hidden_parents=False),
    }
    selected = {k: v for k, v in model_builders.items() if k in models_to_check}
    if not selected:
        raise ValueError(f"No known model in models_to_check={models_to_check}; "
                          f"choose from {list(model_builders)}")

    sample_sizes = [100, 500, 1000]
    feature_sizes = [2, 5, 10]
    strengths = ["weak", "moderate", "strong"]

    rows = []
    for n in sample_sizes:
        for p in feature_sizes:
            for strength in strengths:
                for rep in range(n_reps):
                    rep_seed = make_rep_seed(n, p, "multimodal", strength, rep)
                    X_train, y_train, _ = generate_correlated_data(n, p, "multimodal", strength, rep_seed)
                    test_seed = (rep_seed + _TEST_SEED_OFFSET) % (2 ** 32 - 1)
                    X_test, y_test, _ = generate_correlated_data(TEST_SIZE, p, "multimodal", strength, test_seed)
                    for name, builder in selected.items():
                        try:
                            model = builder().fit(X_train, y_train)
                            y_proba = model.predict_proba(X_test)
                            diag = _reliability_diagram_data(y_test, y_proba, n_bins=n_bins)
                            diag.insert(0, "model", name)
                            diag.insert(0, "rep", rep)
                            diag.insert(0, "strength", strength)
                            diag.insert(0, "p", p)
                            diag.insert(0, "n", n)
                            rows.append(diag)
                        except Exception as e:
                            print(f"    [ERROR] reliability diag {name} "
                                  f"n={n},p={p},{strength},rep={rep}: {e}")

    df_raw = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _weighted_pool(g):
        w = g["n_in_bin"].to_numpy(dtype=float)
        if w.sum() == 0:
            return pd.Series({"pooled_n_in_bin": 0, "pooled_confidence": np.nan,
                               "pooled_accuracy": np.nan, "pooled_abs_gap": np.nan})
        conf = np.nansum(g["mean_confidence"].to_numpy() * w) / w.sum()
        acc = np.nansum(g["empirical_accuracy"].to_numpy() * w) / w.sum()
        return pd.Series({"pooled_n_in_bin": int(w.sum()), "pooled_confidence": conf,
                           "pooled_accuracy": acc, "pooled_abs_gap": float(abs(conf - acc))})

    if not df_raw.empty:
        df_pooled = (
            df_raw.groupby(["n", "p", "strength", "model", "bin_index", "bin_lo", "bin_hi"])
            .apply(_weighted_pool)
            .reset_index()
        )
    else:
        df_pooled = pd.DataFrame()

    return df_raw, df_pooled


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

    df_diagnostics = summarize_diagnostics(df_results)
    diagnostics_file = f"simulation_results_diagnostics_{timestamp}.csv"
    df_diagnostics.to_csv(diagnostics_file, index=False)
    print(f"Scenario diagnostics (realized correlation, bimodality coefficient) saved to '{diagnostics_file}'")

    df_bandwidth_raw, df_bandwidth_per_cell, bandwidth_pooled = run_bivariate_bandwidth_sensitivity_check()
    bandwidth_raw_file = f"bivariate_bandwidth_sensitivity_raw_{timestamp}.csv"
    bandwidth_percell_file = f"bivariate_bandwidth_sensitivity_per_cell_{timestamp}.csv"
    df_bandwidth_raw.to_csv(bandwidth_raw_file, index=False)
    df_bandwidth_per_cell.to_csv(bandwidth_percell_file, index=False)
    print(f"Bivariate bandwidth sensitivity check (p=2, all n, {10} reps/cell) saved to "
          f"'{bandwidth_raw_file}' (raw) and '{bandwidth_percell_file}' (per n/strength).")
    print(f"Pooled relative UCV-score gap (drop into the paper's placeholder sentence): "
          f"mean={bandwidth_pooled['pooled_mean_relative_gap']:.4f}, "
          f"std={bandwidth_pooled['pooled_std_relative_gap']:.4f}, "
          f"n={bandwidth_pooled['pooled_n']}")

    df_reliability_raw, df_reliability_pooled = run_reliability_diagnostics()
    reliability_raw_file = f"reliability_diagnostics_raw_{timestamp}.csv"
    reliability_pooled_file = f"reliability_diagnostics_pooled_{timestamp}.csv"
    df_reliability_raw.to_csv(reliability_raw_file, index=False)
    df_reliability_pooled.to_csv(reliability_pooled_file, index=False)
    print(f"Reliability-diagram / bin-occupancy diagnostics (SHNB K=5 vs HNB 10-bin) "
          f"saved to '{reliability_raw_file}' (raw) and '{reliability_pooled_file}' (pooled).")
