import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import digamma
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin


class SemiparametricHNB(BaseEstimator, ClassifierMixin):
    def __init__(self, k_neighbors=5, ucv_grid_points=20, importance_weight_cap=None,
                 bandwidth_subsample_size=None):
        self.k_neighbors = k_neighbors
        self.ucv_grid_points = ucv_grid_points
        self.importance_weight_cap = importance_weight_cap
        self.bandwidth_subsample_size = bandwidth_subsample_size

    def fit(self, X, y):
        t_start = time.perf_counter()
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y) if not isinstance(y, pd.Series) else y
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        self.feature_types_ = ['continuous'] * self.n_features_
        self._estimate_priors()
        self._estimate_parametric_parameters()
        self._compute_weights()
        self._select_bandwidth()
        self._select_pairwise_bandwidth()
        self._precompute_train_densities()
        self.X_train_groups_ = {c: self.X_train_[self.y_train_ == c].values for c in self.classes_}
        self.fit_time_seconds_ = time.perf_counter() - t_start
        return self

    @classmethod
    def fit_family(cls, X, y, k_values, **shared_kwargs):
        t_shared_start = time.perf_counter()

        base = cls(k_neighbors=k_values[0], **shared_kwargs)
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_s = pd.Series(y) if not isinstance(y, pd.Series) else y

        base.X_train_ = X_df
        base.y_train_ = y_s
        base.classes_ = np.unique(y_s)
        base.n_features_ = X_df.shape[1]
        base.n_samples_ = X_df.shape[0]
        base.feature_types_ = ['continuous'] * base.n_features_

        base._estimate_priors()
        base._estimate_parametric_parameters()
        base._select_bandwidth()
        base._select_pairwise_bandwidth()
        base._precompute_train_densities()
        base.X_train_groups_ = {
            c: base.X_train_[base.y_train_ == c].values for c in base.classes_
        }

        shared_fit_time = time.perf_counter() - t_shared_start

        shared_attrs = (
            'X_train_', 'y_train_', 'classes_', 'n_features_', 'n_samples_',
            'feature_types_', 'class_priors_', 'means_', 'covs_', 'stds_',
            'bandwidths_', 'bandwidths_pairwise_', 'train_marginal_density_',
            'train_joint_density_', '_pair_inv_', '_pair_det_', 'X_train_groups_',
        )

        models = {}
        for k in k_values:
            t_own_start = time.perf_counter()
            m = cls(k_neighbors=k, **shared_kwargs)
            for attr in shared_attrs:
                setattr(m, attr, getattr(base, attr))
            m._compute_weights()  # the only step that actually depends on k
            own_time = time.perf_counter() - t_own_start
            m.fit_time_seconds_ = shared_fit_time / len(k_values) + own_time
            models[k] = m

        return models, shared_fit_time

    def _estimate_priors(self):
        counts = self.y_train_.value_counts()
        self.class_priors_ = {c: (counts.get(c, 0) + 1) / (self.n_samples_ + len(self.classes_)) for c in self.classes_}

    def _estimate_parametric_parameters(self):
        self.means_ = {}; self.covs_ = {}; self.stds_ = {}
        for c in self.classes_:
            Xc = self.X_train_[self.y_train_ == c]
            self.means_[c] = Xc.mean().values
            self.covs_[c] = Xc.cov().values + np.eye(self.n_features_) * 1e-6
            self.stds_[c] = np.sqrt(np.diag(self.covs_[c]))

    def _compute_weights(self):
        d = self.n_features_
        X_std, C_std = self._standardize_for_cmi()
        feature_C_trees = self._build_feature_C_trees(X_std, C_std)
        c_only_tree = cKDTree(C_std.reshape(-1, 1))
        W = np.zeros((d, d))
        for i in range(d):
            for j in range(i + 1, d):
                cmi = self._calculate_cmi(i, j, X_std, C_std, feature_C_trees, c_only_tree)
                W[i, j] = cmi; W[j, i] = cmi
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self.weights_ = W / row_sums
        self.parents_ = {j: [(k, self.weights_[j, k]) for k in range(d) if k != j and self.weights_[j, k] > 1e-6] for j in range(d)}

    def _standardize_for_cmi(self):
        X = self.X_train_.values.astype(float)
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        y = self.y_train_.values
        class_to_idx = {c: idx for idx, c in enumerate(self.classes_)}
        C_num = np.array([class_to_idx[c] for c in y], dtype=float)
        C_std = (C_num - C_num.mean()) / (C_num.std() + 1e-10)
        return X_std, C_std

    def _build_feature_C_trees(self, X_std, C_std):
        d = self.n_features_
        return [cKDTree(np.column_stack([X_std[:, k], C_std])) for k in range(d)]

    def _calculate_cmi(self, i, j, X_std, C_std, feature_C_trees, c_only_tree):
        N = X_std.shape[0]
        K = self.k_neighbors
        if N <= K + 1:
            return 0.0

        Xi_std = X_std[:, i]
        Xj_std = X_std[:, j]

        joint = np.column_stack([Xi_std, Xj_std, C_std])
        tree_joint = cKDTree(joint)
        dist, _ = tree_joint.query(joint, k=K + 1)
        eps = dist[:, K]

        AiC = np.column_stack([Xi_std, C_std])
        AjC = np.column_stack([Xj_std, C_std])
        Cc = C_std.reshape(-1, 1)

        tree_AiC = feature_C_trees[i]
        tree_AjC = feature_C_trees[j]
        tree_C = c_only_tree

        cnt_AiC = tree_AiC.query_ball_point(AiC, r=eps, return_length=True)
        cnt_AjC = tree_AjC.query_ball_point(AjC, r=eps, return_length=True)
        cnt_C = tree_C.query_ball_point(Cc, r=eps, return_length=True)

        eta_AiC = np.maximum(cnt_AiC - 1, 1)  # -1 excludes self (distance 0)
        eta_AjC = np.maximum(cnt_AjC - 1, 1)
        eta_C = np.maximum(cnt_C - 1, 1)

        cmi = digamma(K) - np.mean(digamma(eta_AiC) + digamma(eta_AjC) - digamma(eta_C))
        return max(float(cmi), 0.0)

    def _bandwidth_subsample(self, Xc_full):
        cap = self.bandwidth_subsample_size
        if cap is None or Xc_full.shape[0] <= cap:
            return Xc_full
        rng = np.random.RandomState(0)
        idx = rng.choice(Xc_full.shape[0], size=cap, replace=False)
        return Xc_full[idx]

    def _select_bandwidth(self):
        self.bandwidths_ = {}

        for c in self.classes_:
            Xc_full = self.X_train_[self.y_train_ == c].values
            Xc = self._bandwidth_subsample(Xc_full)
            n, d = Xc.shape

            std_devs = np.std(Xc, axis=0, ddof=1)
            degenerate = std_devs < 1e-10
            safe_std = np.where(degenerate, 1.0, std_devs)

            diff2 = (Xc[:, None, :] - Xc[None, :, :]) ** 2  # (n, n, d)
            diag_idx = np.arange(n)

            r_grid = np.linspace(0.01, 2.0, self.ucv_grid_points)
            best_h = np.zeros(d)
            best_score = np.full(d, np.inf)

            for r in r_grid:
                h = r * safe_std
                h2 = h ** 2

                K2 = np.exp(-diff2 / (4 * h2)[None, None, :])
                term1 = K2.sum(axis=(0, 1)) / (n ** 2 * h * np.sqrt(4 * np.pi))

                K = np.exp(-diff2 / (2 * h2)[None, None, :])
                K[diag_idx, diag_idx, :] = 0.0
                f_loo = K.sum(axis=1) / ((n - 1) * h * np.sqrt(2 * np.pi))
                term2 = 2 * f_loo.mean(axis=0)

                score = term1 - term2
                improve = score < best_score
                best_score = np.where(improve, score, best_score)
                best_h = np.where(improve, h, best_h)

            best_h = np.maximum(best_h, 1e-3 * safe_std)
            best_h = np.where(degenerate, 1.0, best_h)  # match original: no floor for degenerate dims
            self.bandwidths_[c] = best_h

    def _select_pairwise_bandwidth(self):
        self.bandwidths_pairwise_ = {}
        d = self.n_features_
        for c in self.classes_:
            h1d = self.bandwidths_[c]
            self.bandwidths_pairwise_[c] = {
                (i, j): (h1d[i], h1d[j])
                for i in range(d) for j in range(i + 1, d)
            }

    def _precompute_train_densities(self):
        d = self.n_features_
        self.train_marginal_density_ = {}; self.train_joint_density_ = {}
        self._pair_inv_ = {}; self._pair_det_ = {}
        for c in self.classes_:
            Xc = self.X_train_[self.y_train_ == c].values
            mu = self.means_[c]; sigma = self.stds_[c]
            f_marg = norm.pdf(Xc, loc=mu, scale=sigma)
            self.train_marginal_density_[c] = np.maximum(f_marg, 1e-15)
            self.train_joint_density_[c] = {}; self._pair_inv_[c] = {}; self._pair_det_[c] = {}
            for i in range(d):
                for j in range(i + 1, d):
                    mu_ij = mu[[i, j]]
                    cov_ij = self.covs_[c][np.ix_([i, j], [i, j])]
                    det = cov_ij[0, 0] * cov_ij[1, 1] - cov_ij[0, 1] * cov_ij[1, 0]
                    det = max(det, 1e-15)
                    inv = np.array([[cov_ij[1, 1], -cov_ij[0, 1]], [-cov_ij[1, 0], cov_ij[0, 0]]]) / det
                    diff = Xc[:, [i, j]] - mu_ij
                    exponent = -0.5 * np.einsum('ni,ij,nj->n', diff, inv, diff)
                    f_joint = np.exp(exponent) / (2 * np.pi * np.sqrt(det))
                    self.train_joint_density_[c][(i, j)] = np.maximum(f_joint, 1e-15)
                    self._pair_inv_[c][(i, j)] = inv
                    self._pair_det_[c][(i, j)] = det

    def _batch_marginal_density(self, X, c, Xc):
        mu = self.means_[c]; sigma = self.stds_[c]; h = self.bandwidths_[c]
        d = self.n_features_; n_test = X.shape[0]
        f_param_x_test = np.maximum(norm.pdf(X, loc=mu, scale=sigma), 1e-15)
        f_param_Xtrain = self.train_marginal_density_[c]
        marginal = np.empty((n_test, d))
        for j in range(d):
            u = (Xc[:, j][None, :] - X[:, j][:, None]) / h[j]
            K_h = np.exp(-0.5 * u ** 2) / (np.sqrt(2 * np.pi) * h[j])
            inv_f_train = 1.0 / f_param_Xtrain[:, j][None, :]
            if self.importance_weight_cap is not None:
                inv_f_train = np.minimum(inv_f_train, self.importance_weight_cap)
            correction = np.mean(K_h * inv_f_train, axis=1)
            marginal[:, j] = f_param_x_test[:, j] * np.maximum(correction, 1e-10)
        return np.maximum(marginal, 1e-15)

    def _batch_joint_density(self, X, c, Xc, pairs):
        mu = self.means_[c]; joint = {}
        for (i, j) in pairs:
            mu_ij = mu[[i, j]]
            inv = self._pair_inv_[c][(i, j)]; det = self._pair_det_[c][(i, j)]
            diff_test = X[:, [i, j]] - mu_ij
            exponent_test = -0.5 * np.einsum('ni,ij,nj->n', diff_test, inv, diff_test)
            f_param_x_test = np.maximum(np.exp(exponent_test) / (2 * np.pi * np.sqrt(det)), 1e-15)
            f_param_Xtrain = self.train_joint_density_[c][(i, j)]
            h_i, h_j = self.bandwidths_pairwise_[c][(i, j)]
            diff_i = (Xc[:, i][None, :] - X[:, i][:, None]) / h_i
            diff_j = (Xc[:, j][None, :] - X[:, j][:, None]) / h_j
            K_joint = np.exp(-0.5 * (diff_i ** 2 + diff_j ** 2)) / (2 * np.pi * h_i * h_j)
            inv_f_train = 1.0 / f_param_Xtrain[None, :]
            if self.importance_weight_cap is not None:
                inv_f_train = np.minimum(inv_f_train, self.importance_weight_cap)
            correction = np.mean(K_joint * inv_f_train, axis=1)
            density = f_param_x_test * np.maximum(correction, 1e-10)
            joint[(i, j)] = np.maximum(density, 1e-15)
        return joint

    def predict_proba(self, X, batch_size=512):
        t_start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        n_test = X.shape[0]; d = self.n_features_; n_classes = len(self.classes_)
        if not hasattr(self, 'train_marginal_density_'): self._precompute_train_densities()
        if not hasattr(self, 'X_train_groups_'):
            self.X_train_groups_ = {c: self.X_train_[self.y_train_ == c].values for c in self.classes_}
        if not hasattr(self, 'parents_'):
            self.parents_ = {j: [(k, self.weights_[j, k]) for k in range(d) if k != j and self.weights_[j, k] > 1e-6] for j in range(d)}
        needed_pairs = [(i, j) for i in range(d) for j in range(i + 1, d) if self.weights_[i, j] > 1e-6 or self.weights_[j, i] > 1e-6]
        if batch_size is None: batch_size = n_test
        batch_size = max(int(batch_size), 1)
        probs = np.empty((n_test, n_classes))
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            X_batch = X[start:end]; n_batch = X_batch.shape[0]
            log_probs = np.zeros((n_batch, n_classes))
            for c_idx, c in enumerate(self.classes_):
                Xc = self.X_train_groups_[c]
                marginal_density = self._batch_marginal_density(X_batch, c, Xc)
                joint_density = self._batch_joint_density(X_batch, c, Xc, needed_pairs)
                log_prob_c = np.full(n_batch, np.log(self.class_priors_[c]))
                for j in range(d):
                    hidden_parent_prob = np.zeros(n_batch); weight_sum = 0.0
                    for k, w_jk in self.parents_[j]:
                        pair_key = (j, k) if j < k else (k, j)
                        joint = joint_density.get(pair_key)
                        if joint is None: continue
                        cond_prob = joint / marginal_density[:, k]
                        hidden_parent_prob += w_jk * cond_prob; weight_sum += w_jk
                    if weight_sum == 0: hidden_parent_prob = marginal_density[:, j]
                    else: hidden_parent_prob = hidden_parent_prob / weight_sum
                    log_prob_c += np.log(np.maximum(hidden_parent_prob, 1e-15))
                log_probs[:, c_idx] = log_prob_c
            max_log = log_probs.max(axis=1, keepdims=True)
            exps = np.exp(log_probs - max_log)
            probs[start:end] = exps / exps.sum(axis=1, keepdims=True)
        elapsed = time.perf_counter() - t_start
        self.last_predict_time_seconds_ = elapsed
        self.last_predict_n_points_ = n_test
        self.last_predict_seconds_per_point_ = elapsed / max(n_test, 1)
        return probs

    def predict(self, X, batch_size=512):
        probs = self.predict_proba(X, batch_size=batch_size)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y, batch_size=512):
        predictions = self.predict(X, batch_size=batch_size)
        return np.mean(predictions == y)
