import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from itertools import product
from collections import defaultdict

class Hidden_NB():
    """
    Hidden_NB implements the Hidden Naive Bayes Classifer from
    "A Novel Bayes Model: Hidden Naive Bayes" by Jiang, Zhang, and Chai.

    This implementation automatically discretizes continuous features using
    equal-width binning while preserving categorical features.
    """

    def __init__(self, n_bins=10):
        """
        Parameters:
        -----------
        n_bins : int, default=10
            Number of bins for discretizing continuous features
        """
        self.n_bins = n_bins
        self.discretizers_ = {}
        self.continuous_features_ = []
        self.categorical_features_ = []


    def fit(self, X, y):
        # Convert to DataFrame if necessary
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        X = X.copy()

        # Identify different types of features
        self._identify_feature_types(X)

        # Discretize continuous features
        X_processed = self._discretize_features(X, fit=True)

        self._set_attributes(X_processed, y)

        # Calculate conditionals p(a_i, a_c, c) for each attribute pair
        predictor_names = product(X_processed.columns, repeat=2)
        self._attribute_pair_conditionals = defaultdict(lambda: defaultdict(dict))
        for Ai, Aj in predictor_names:
            for c in self.classes:
                self._attribute_pair_conditionals[c][Ai][Aj] = self._generate_attribute_pair_conditional(Ai, Aj, c)

        self._calculate_weights()

        return self


    def predict(self, X):
        # Convert to DataFrame if necessary
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        X = X.copy()

        # Discretize using fitted discretizers
        X_processed = self._discretize_features(X, fit=False)

        return X_processed.apply(self._classify_record, axis=1)


    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Returns:
        --------
        proba : array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        # Convert to DataFrame if necessary
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        X = X.copy()
        
        # Discretize using fitted discretizers
        X_processed = self._discretize_features(X, fit=False)
        
        # Get probabilities for each sample
        probas = []
        for _, record in X_processed.iterrows():
            class_probs = {}
            for c in self.classes:
                hidden_parents = np.array([self._generate_hidden_parent_prob(record, Ai, ai, c) 
                                          for Ai, ai in record.items()])
                class_probs[c] = np.exp(np.sum(np.log(hidden_parents)) + np.log(self._p_c[c]))
            
            # Normalize to get probabilities
            total = sum(class_probs.values())
            if total == 0:
                total = 1e-10
            normalized_probs = {c: prob/total for c, prob in class_probs.items()}
            
            # Convert to array in order of self.classes
            prob_array = [normalized_probs[c] for c in sorted(self.classes)]
            probas.append(prob_array)
        
        return np.array(probas)


    def _identify_feature_types(self, X):
        """Identify which features are continuous vs categorical"""
        self.continuous_features_ = []
        self.categorical_features_ = []

        for col in X.columns:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(X[col]):
                if X[col].dtype == float or X[col].nunique() > 10:
                    self.continuous_features_.append(col)
                else:
                    self.categorical_features_.append(col)
            else:
                self.categorical_features_.append(col)


    def _discretize_features(self, X, fit=True):
        X_processed = X.copy()

        # Discretize continuous features
        for col in self.continuous_features_:
            if fit:
                discretizer = KBinsDiscretizer(
                    n_bins=self.n_bins,
                    encode='ordinal',
                    strategy='uniform'
                )
                X_processed[col] = discretizer.fit_transform(X[[col]]).flatten()
                self.discretizers_[col] = discretizer
            else:
                if col in self.discretizers_:
                    X_processed[col] = self.discretizers_[col].transform(X[[col]]).flatten()

        X_processed = X_processed.astype(str)

        return X_processed


    def _set_attributes(self, X, y):
        self.classes = y.unique()
        self._n = X.shape[0]
        self._p_c = (y.value_counts()+1.0)/(self._n + 1.0)
        self.predictors = X
        self.target = y


    def _classify_record(self, record):
        classifications = {}
        for c in self.classes:
            hidden_parents = np.array([self._generate_hidden_parent_prob(record, Ai, ai, c)
                                      for Ai, ai in record.items()])
            classifications[c] = np.sum(np.log(hidden_parents))+np.log(self._p_c[c])

        return pd.Series(classifications).idxmax()


    def _generate_hidden_parent_prob(self, record, Ai, ai, c):
        class_conditionals = self._attribute_pair_conditionals[c]
        attribute_conditionals = class_conditionals[Ai]

        hidden_parent = 0
        for Aj, aj in record.items():
            conditional = attribute_conditionals[Aj]

            if ((aj in conditional.index) and (ai in conditional.columns)):
                hidden_parent += self.weights.loc[Ai,Aj]*attribute_conditionals[Aj].loc[aj, ai]

        return hidden_parent + 1e-12


    def _normalize(self, predictions):
        normalizer = sum(predictions.values())

        for c in predictions.keys():
            predictions[c] = predictions[c]/normalizer

        return predictions


    def _generate_attribute_pair_conditional(self, Ai, Aj, c):
        # Calculates conditional for single pair of attributes
        Ai_series = self.predictors.loc[self.target==c, Ai]
        Aj_series = self.predictors.loc[self.target==c, Aj]
        Ai_Aj = self.predictors.loc[self.target==c, [Ai, Aj]]

        smoother = 1.0/Ai_series.nunique()

        if (Ai==Aj):
            counts = Ai_series.value_counts()
            smooth_crosstab = pd.DataFrame(0, index=counts.index, columns=counts.index)

            for i, count in counts.items():
                smooth_crosstab.loc[i,i] = count
            smooth_crosstab = smooth_crosstab + smoother

        else:
            smooth_crosstab = Ai_Aj.groupby([Aj, Ai]).size().unstack(Ai, fill_value=0) + smoother

        normalizer = Aj_series.value_counts(sort=False) + 1.0
        conditionals = smooth_crosstab.divide(normalizer, axis=0)

        return conditionals


    def _conditional_MI(self, Ai, Aj):
        # Calculates conditional mutual information for a single pair of attributes
        CMI = 0
        if (Ai==Aj):
            return CMI
        else:
            for c in self.classes:
                Ai_Aj = self.predictors.loc[self.target==c, [Ai, Aj]]
                CMI = CMI + mutual_info_score(Ai_Aj[Ai].astype('str'), Ai_Aj[Aj].astype('str'))

        return CMI


    def _all_conditional_MI(self):
        # Calculates conditional mutual information for all attributes
        all_CMI_array = [[self._conditional_MI(Ai, Aj) for Ai in self.predictors.columns]
                         for Aj in self.predictors.columns]
        all_CMI = pd.DataFrame(all_CMI_array, index=self.predictors.columns, columns=self.predictors.columns)

        return all_CMI


    def _calculate_weights(self):
        all_CMI = self._all_conditional_MI()
        weights = all_CMI.div(all_CMI.sum())
        self.weights = weights
