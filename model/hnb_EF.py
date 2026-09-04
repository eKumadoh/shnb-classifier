import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from itertools import product
from collections import defaultdict


class Hidden_NB_EqualFrequency():
    
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.discretizers_ = {}
        self.continuous_features_ = []
        self.categorical_features_ = []
        self.n_categories_ = {}

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        X = X.copy()

        self._identify_feature_types(X)
        X_processed = self._discretize_features(X, fit=True)
        self._set_attributes(X_processed, y)

        predictor_names = product(X_processed.columns, repeat=2)
        self._attribute_pair_conditionals = defaultdict(lambda: defaultdict(dict))
        for Ai, Aj in predictor_names:
            for c in self.classes:
                self._attribute_pair_conditionals[c][Ai][Aj] = self._generate_attribute_pair_conditional(Ai, Aj, c)

        self._calculate_weights()

        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        X = X.copy()
        X_processed = self._discretize_features(X, fit=False)

        return X_processed.apply(self._classify_record, axis=1)

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        X = X.copy()
        X_processed = self._discretize_features(X, fit=False)

        probas = []
        for _, record in X_processed.iterrows():
            class_probs = {}
            for c in self.classes:
                hidden_parents = np.array([self._generate_hidden_parent_prob(record, Ai, ai, c)
                                            for Ai, ai in record.items()])
                class_probs[c] = np.exp(np.sum(np.log(hidden_parents)) + np.log(self._p_c[c]))

            total = sum(class_probs.values())
            if total == 0:
                total = 1e-10
            normalized_probs = {c: prob / total for c, prob in class_probs.items()}

            prob_array = [normalized_probs[c] for c in sorted(self.classes)]
            probas.append(prob_array)

        return np.array(probas)

    def _identify_feature_types(self, X):
        self.continuous_features_ = []
        self.categorical_features_ = []

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                if X[col].dtype == float or X[col].nunique() > 10:
                    self.continuous_features_.append(col)
                else:
                    self.categorical_features_.append(col)
            else:
                self.categorical_features_.append(col)

    def _discretize_features(self, X, fit=True):
        X_processed = X.copy()

        for col in self.continuous_features_:
            if fit:
                try:
                    discretizer = KBinsDiscretizer(
                        n_bins=self.n_bins, encode='ordinal', strategy='quantile',
                        quantile_method='linear'
                    )
                    X_processed[col] = discretizer.fit_transform(X[[col]]).flatten()
                except TypeError:
                    discretizer = KBinsDiscretizer(
                        n_bins=self.n_bins, encode='ordinal', strategy='quantile'
                    )
                    X_processed[col] = discretizer.fit_transform(X[[col]]).flatten()
                self.discretizers_[col] = discretizer
                self.n_categories_[col] = int(discretizer.n_bins_[0])
            else:
                if col in self.discretizers_:
                    X_processed[col] = self.discretizers_[col].transform(X[[col]]).flatten()

        X_processed = X_processed.astype(str)

        return X_processed

    def _set_attributes(self, X, y):
        self.classes = y.unique()
        self._n = X.shape[0]
        k = len(self.classes)
        self._p_c = (y.value_counts() + 1.0) / (self._n + k)
        self.predictors = X
        self.target = y

    def _classify_record(self, record):
        classifications = {}
        for c in self.classes:
            hidden_parents = np.array([self._generate_hidden_parent_prob(record, Ai, ai, c)
                                        for Ai, ai in record.items()])
            classifications[c] = np.sum(np.log(hidden_parents)) + np.log(self._p_c[c])

        return pd.Series(classifications).idxmax()

    def _generate_hidden_parent_prob(self, record, Ai, ai, c):
        class_conditionals = self._attribute_pair_conditionals[c]
        attribute_conditionals = class_conditionals[Ai]

        hidden_parent = 0
        for Aj, aj in record.items():
            conditional = attribute_conditionals[Aj]

            if ((aj in conditional.index) and (ai in conditional.columns)):
                hidden_parent += self.weights.loc[Ai, Aj] * attribute_conditionals[Aj].loc[aj, ai]

        return hidden_parent + 1e-12

    def _normalize(self, predictions):
        normalizer = sum(predictions.values())

        for c in predictions.keys():
            predictions[c] = predictions[c] / normalizer

        return predictions

    def _full_labels(self, col):
        """Full set of nu_i possible bin labels for `col` (see hnb.py for
        rationale; identical convention used here for the equal-frequency
        variant)."""
        if col in self.n_categories_:
            return [str(float(i)) for i in range(self.n_categories_[col])]
        return sorted(self.predictors[col].unique())

    def _generate_attribute_pair_conditional(self, Ai, Aj, c):
        Ai_series = self.predictors.loc[self.target == c, Ai]
        Aj_series = self.predictors.loc[self.target == c, Aj]
        Ai_Aj = self.predictors.loc[self.target == c, [Ai, Aj]]

        all_Ai_labels = self._full_labels(Ai)
        all_Aj_labels = self._full_labels(Aj)
        nu_i = len(all_Ai_labels)  # Eq. (9): nu_i = number of possible categories of A_i

        if Ai == Aj:
            counts = Ai_series.value_counts().reindex(all_Ai_labels, fill_value=0)
            smooth_crosstab = pd.DataFrame(0.0, index=all_Ai_labels, columns=all_Ai_labels)
            for label, count in counts.items():
                smooth_crosstab.loc[label, label] = count
            smooth_crosstab = smooth_crosstab + 1.0  # Eq. (9) numerator: n_ijc + 1
        else:
            smooth_crosstab = (
                Ai_Aj.groupby([Aj, Ai]).size()
                .unstack(Ai, fill_value=0)
                .reindex(index=all_Aj_labels, columns=all_Ai_labels, fill_value=0)
                + 1.0  
            )

        normalizer = Aj_series.value_counts().reindex(all_Aj_labels, fill_value=0) + nu_i
        conditionals = smooth_crosstab.divide(normalizer, axis=0)
        return conditionals

    def _conditional_MI(self, Ai, Aj):
        if (Ai == Aj):
            return 0.0
        CMI = 0.0
        n_total = len(self.target)
        for c in self.classes:
            mask = self.target == c
            p_c = mask.sum() / n_total
            Ai_Aj = self.predictors.loc[mask, [Ai, Aj]]
            CMI += p_c * mutual_info_score(Ai_Aj[Ai].astype('str'), Ai_Aj[Aj].astype('str'))
        return CMI

    def _all_conditional_MI(self):
        all_CMI_array = [[self._conditional_MI(Ai, Aj) for Ai in self.predictors.columns]
                          for Aj in self.predictors.columns]
        all_CMI = pd.DataFrame(all_CMI_array, index=self.predictors.columns, columns=self.predictors.columns)

        return all_CMI

    def _calculate_weights(self):
        all_CMI = self._all_conditional_MI()
        row_sums = all_CMI.sum(axis=1).replace(0, 1.0)
        weights = all_CMI.div(row_sums, axis=0)
        self.weights = weights
