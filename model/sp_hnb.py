import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
from scipy.special import digamma, gammaln
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin

class SemiparametricHNB(BaseEstimator, ClassifierMixin):
    def __init__(self, k_neighbors=5, ucv_grid_points=20):
        """
        Semiparametric Hidden Naive Bayes Classifier

        Parameters:
        -----------
        k_neighbors : int
            k for the k-NN entropy estimation (Kozachenko-Leonenko estimator)
        bandwidth_method : str
            Method for bandwidth selection: 'ucv'
        ucv_grid_points : int
            Number of grid points for UCV optimization
        """
        self.k_neighbors = k_neighbors
        self.ucv_grid_points = ucv_grid_points

    def fit(self, X, y):
        """Fit the Semiparametric HNB model"""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y) if not isinstance(y, pd.Series) else y

        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]

        self.feature_types_ = ['continuous'] * self.n_features_

        # 1. Estimate Class Priors with Laplace smoothing
        self._estimate_priors()

        # 2. Estimate Parametric Parameters (Gaussian Start)
        self._estimate_parametric_parameters()

        # 3. Compute Conditional Mutual Information Weights (Equation 3.14)
        self._compute_weights()

        # 4. Select Optimal Bandwidths
        self._select_bandwidth()

        return self

    def _estimate_priors(self):
        """Estimate class priors with Laplace smoothing"""
        counts = self.y_train_.value_counts()
        self.class_priors_ = {
            c: (counts.get(c, 0) + 1) / (self.n_samples_ + len(self.classes_))
            for c in self.classes_
        }

    def _estimate_parametric_parameters(self):
        """
        Estimate mean vectors and covariance matrices for the parametric
        Gaussian start f(x, θ)
        """
        self.means_ = {}
        self.covs_ = {}
        self.stds_ = {}

        for c in self.classes_:
            Xc = self.X_train_[self.y_train_ == c]
            self.means_[c] = Xc.mean().values
            # Regularization to prevent singular matrices
            self.covs_[c] = Xc.cov().values + np.eye(self.n_features_) * 1e-6
            self.stds_[c] = np.sqrt(np.diag(self.covs_[c]))

    def _compute_weights(self):
        """
        Compute weights w_ij based on Conditional Mutual Information (CMI)
        using k-NN estimator

        I(Ai; Aj | C) = H(Ai,C) + H(Aj,C) - H(C) - H(Ai,Aj,C)
        """
        d = self.n_features_
        W = np.zeros((d, d))

        for i in range(d):
            for j in range(d):
                if i != j:
                    W[i, j] = self._calculate_cmi(i, j)

        # Normalize weights: w_ij = I(Ai;Aj|C) / sum_l I(Ai;Al|C)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        self.weights_ = W / row_sums

    def _calculate_cmi(self, i, j):
        """
        Calculate I(Ai; Aj | C) using k-NN entropy estimator

        I(Ai; Aj | C) = H(Ai,C) + H(Aj,C) - H(C) - H(Ai,Aj,C)
        """
        y_numeric = pd.Categorical(self.y_train_).codes.reshape(-1, 1)

        Xi = self.X_train_.iloc[:, i].values.reshape(-1, 1)
        Xj = self.X_train_.iloc[:, j].values.reshape(-1, 1)

        # Standardize features for entropy estimation
        Xi_std = (Xi - Xi.mean()) / (Xi.std() + 1e-10)
        Xj_std = (Xj - Xj.mean()) / (Xj.std() + 1e-10)

        C = y_numeric / (len(self.classes_) - 1) if len(self.classes_) > 1 else y_numeric

        # Compute entropies
        H_AiC = self._knn_entropy(np.hstack([Xi_std, C]))
        H_AjC = self._knn_entropy(np.hstack([Xj_std, C]))
        H_C = self._knn_entropy(C)
        H_AiAjC = self._knn_entropy(np.hstack([Xi_std, Xj_std, C]))

        cmi = H_AiC + H_AjC - H_C - H_AiAjC

        return max(cmi, 0.0)  

    def _knn_entropy(self, X):
        """
        Kozachenko-Leonenko k-NN entropy estimator
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n, d = X.shape

        if n <= self.k_neighbors:
            return 0.0

        # Fit k-NN
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1,
                               algorithm='auto').fit(X)
        distances, _ = nbrs.kneighbors(X)

        # Distance to k-th neighbor
        eps = distances[:, self.k_neighbors]

        # Avoid log(0)
        eps = np.maximum(eps, 1e-10)

        # Volume of unit ball in d dimensions: c_d = π^(d/2) / Γ(d/2 + 1)
        cd = (np.pi ** (d / 2)) / np.exp(gammaln(d / 2 + 1))

        entropy = digamma(n) - digamma(self.k_neighbors) + np.log(cd) + \
                  (d / n) * np.sum(np.log(eps))

        return entropy

    def _select_bandwidth(self):
        """
        Bandwidth selection using Unbiased Cross Validation
        """
        self.bandwidths_ = {}

        for c in self.classes_:
            Xc = self.X_train_[self.y_train_ == c].values
            n, d = Xc.shape
            self.bandwidths_[c] = np.zeros(d)

            for dim in range(d):
                data = Xc[:, dim]
                std_dev = np.std(data, ddof=1)

                if std_dev < 1e-10:
                    self.bandwidths_[c][dim] = 1.0
                    continue

                h = self._ucv_bandwidth(data, std_dev)
                self.bandwidths_[c][dim] = max(h, 1e-3 * std_dev)


    def _ucv_bandwidth(self, data, std_dev):
        """
        Unbiased Cross-Validation bandwidth selection
        """
        data = data.reshape(-1, 1)
        n = len(data)

        diff2 = (data - data.T) ** 2
        h_min = 0.1 * std_dev
        h_max = 2.0 * std_dev
        h_grid = np.linspace(h_min, h_max, self.ucv_grid_points)
        best_h = h_grid[0]
        best_score = np.inf

        for h in h_grid:
          h2 = h * h
          K2 = np.exp(-diff2 / (4 * h2))
          term1 = K2.sum() / (n**2 * h * np.sqrt(4 * np.pi))

          K = np.exp(-diff2 / (2 * h2))
          np.fill_diagonal(K, 0.0)
          f_loo = K.sum(axis=1) / ((n - 1) * h * np.sqrt(2 * np.pi))
          term2 = 2 * np.mean(f_loo)

          ucv_score = term1 - term2

          if ucv_score < best_score:
            best_score = ucv_score
            best_h = h

        best_h = max(best_h, 1e-3 * std_dev)
        return best_h

    def _semiparametric_density_1d(self, x_val, feat_idx, c, X_c_data):
        h = self.bandwidths_[c][feat_idx]
        mu = self.means_[c][feat_idx]
        sigma = self.stds_[c][feat_idx]

        # Parametric start: f(x, θ̂) - Gaussian (Equation 3.16)
        f_param_x = norm.pdf(x_val, loc=mu, scale=sigma)
        f_param_x = max(f_param_x, 1e-15)

        # Training data for this feature/class
        Xi = X_c_data[:, feat_idx]
        nc = len(Xi)

        # Parametric start on training data
        f_param_Xi = norm.pdf(Xi, loc=mu, scale=sigma)
        f_param_Xi = np.maximum(f_param_Xi, 1e-15)

        # Gaussian kernel: K_h(u) = (1/h√(2π)) * exp(-u²/2h²)
        u = (Xi - x_val) / h
        K_h = (1 / (np.sqrt(2 * np.pi) * h)) * np.exp(-0.5 * u**2)

        # Correction factor r̂(x) (Equation 3.17)
        # r̂(x) = (1/n) Σ K_h(X_i - x) * [f_param(x) / f_param(X_i)]
        correction = np.mean(K_h * (f_param_x / f_param_Xi))

        # Final semiparametric density (Equation 3.15)
        density = f_param_x * max(correction, 1e-10)

        return max(density, 1e-15)

    def _semiparametric_density_2d(self, x_i, x_j, i, j, c, X_c_data):
        """
        Joint semiparametric density for bivariate case
        """
        # Extract bivariate parameters
        mu_vec = self.means_[c][[i, j]]
        cov_mat = self.covs_[c][np.ix_([i, j], [i, j])]

        # Parametric start: Bivariate Gaussian
        x_vec = np.array([x_i, x_j])
        f_param_x = multivariate_normal.pdf(x_vec, mean=mu_vec, cov=cov_mat)
        f_param_x = max(f_param_x, 1e-15)

        # Training data subset
        Xi_data = X_c_data[:, [i, j]]
        nc = len(Xi_data)

        # f_param on training data
        f_param_Xi = multivariate_normal.pdf(Xi_data, mean=mu_vec, cov=cov_mat)
        f_param_Xi = np.maximum(f_param_Xi, 1e-15)

        # Product kernel
        h_i = self.bandwidths_[c][i]
        h_j = self.bandwidths_[c][j]

        diff_i = (Xi_data[:, 0] - x_i) / h_i
        diff_j = (Xi_data[:, 1] - x_j) / h_j

        K_i = (1 / (np.sqrt(2 * np.pi) * h_i)) * np.exp(-0.5 * diff_i**2)
        K_j = (1 / (np.sqrt(2 * np.pi) * h_j)) * np.exp(-0.5 * diff_j**2)
        K_joint = K_i * K_j

        # Correction factor r̂(x_i, x_j | C=c)
        correction = np.mean(K_joint * (f_param_x / f_param_Xi))

        # Final density
        density = f_param_x * max(correction, 1e-10)

        return max(density, 1e-15)

    def _conditional_prob(self, x_i, x_j, i, j, c, X_c_data):
        """
        Calculate P(A_i | A_j, C=c)
        """
        joint_prob = self._semiparametric_density_2d(x_i, x_j, i, j, c, X_c_data)
        marginal_prob_j = self._semiparametric_density_1d(x_j, j, c, X_c_data)

        # Avoid division by zero
        marginal_prob_j = max(marginal_prob_j, 1e-15)

        return joint_prob / marginal_prob_j

    def predict_proba(self, X):
        """
        Predict class probabilities
        """
        X = np.array(X)
        n_test = len(X)
        probs = np.zeros((n_test, len(self.classes_)))

        # Cache training data by class
        X_train_groups = {
            c: self.X_train_[self.y_train_ == c].values
            for c in self.classes_
        }

        for idx, x_row in enumerate(X):
            log_scores = {}

            for c_idx, c in enumerate(self.classes_):
                # Start with log prior: log P(C=c)
                log_prob = np.log(self.class_priors_[c])

                X_c_data = X_train_groups[c]

                for j in range(self.n_features_):
                    term_j = 0.0

                    top_k = 3
                    parent_idx = np.argsort(self.weights_[j])[::-1][:top_k]
                    significant_parents = parent_idx[self.weights_[j, parent_idx] > 1e-6]

                    if len(significant_parents) == 0:
                        marginal_prob = self._semiparametric_density_1d(
                            x_row[j], j, c, X_c_data
                        )
                        term_j = np.log(marginal_prob)
                    else:
                        for k in significant_parents:
                            w_jk = self.weights_[j, k]
                            cond_prob = self._conditional_prob(
                                x_row[j], x_row[k], j, k, c, X_c_data
                            )
                            term_j += w_jk * np.log(max(cond_prob, 1e-15))

                    log_prob += term_j

                log_scores[c] = log_prob

            # Normalizing for numerical stability
            max_log = max(log_scores.values())
            exps = {c: np.exp(val - max_log) for c, val in log_scores.items()}
            sum_exps = sum(exps.values())

            for c_idx, c in enumerate(self.classes_):
                probs[idx, c_idx] = exps[c] / sum_exps

        return probs

    def predict(self, X):
        """Predict class labels"""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        """Return the mean accuracy on the given test data"""
        predictions = self.predict(X)
        return np.mean(predictions == y)