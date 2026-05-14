import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, ClassifierMixin

class KDENaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes_ = None
        self.models_ = {}
        self.priors_ = {}

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.models_ = {}
        self.priors_ = {}

        n_total = len(X)

        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = len(Xc) / n_total
            self.models_[c] = []

            for j in range(X.shape[1]):
                std = np.std(Xc[:, j])
                bw = std * (len(Xc) ** (-1/5)) if std > 0 else 1.0
                kde = KernelDensity(kernel='gaussian', bandwidth=max(bw, 1e-3))
                kde.fit(Xc[:, j:j+1])
                self.models_[c].append(kde)

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, len(self.classes_)))

        for i, x in enumerate(X):
            log_scores = []

            for c in self.classes_:
                log_prob = np.log(self.priors_[c])

                for j, kde in enumerate(self.models_[c]):
                    log_prob += kde.score_samples([[x[j]]])[0]

                log_scores.append(log_prob)

            # Log-sum-exp trick for numerical stability
            max_log = np.max(log_scores)
            exp_scores = np.exp(np.array(log_scores) - max_log)
            probs[i, :] = exp_scores / np.sum(exp_scores)

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
