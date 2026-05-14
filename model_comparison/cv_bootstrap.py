import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, log_loss
)

def multi_cross_val(data, target, estimator, cv=5, n=6):
    
    scores = {
        'precision': np.array([]),
        'recall': np.array([]),
        'f1': np.array([]),
        'accuracy': np.array([]),
        'misclassification_error': np.array([]),
        'auc': np.array([]),
        'log_loss': np.array([])
    }
    
    for _ in range(n):
        kf = KFold(n_splits=cv, shuffle=True)
        
        for train_index, test_index in kf.split(data):
            
            train = data.loc[train_index]
            test = data.loc[test_index]
            
            X_train = train.drop(columns=[target])
            y_train = train[target]
            X_test = test.drop(columns=[target])
            y_test = test[target]
            
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            
            # Some models (rare) may not support predict_proba
            if hasattr(estimator, "predict_proba"):
                y_prob = estimator.predict_proba(X_test)
            else:
                y_prob = None

            precision = precision_score(y_test, y_pred, average='micro')
            recall = recall_score(y_test, y_pred, average='micro')
            f1 = f1_score(y_test, y_pred, average='micro')
            accuracy = accuracy_score(y_test, y_pred)
            mis_error = 1 - accuracy

            # --- AUC ---
            try:
                if y_prob is not None:
                    if len(np.unique(y_test)) == 2:
                        auc = roc_auc_score(y_test, y_prob[:, 1])
                    else:
                        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                else:
                    auc = np.nan
            except:
                auc = np.nan

            # --- Log Loss ---
            try:
                if y_prob is not None:
                    ll = log_loss(y_test, y_prob)
                else:
                    ll = np.nan
            except:
                ll = np.nan

            scores['precision'] = np.append(scores['precision'], precision)
            scores['recall'] = np.append(scores['recall'], recall)
            scores['f1'] = np.append(scores['f1'], f1)
            scores['accuracy'] = np.append(scores['accuracy'], accuracy)
            scores['misclassification_error'] = np.append(scores['misclassification_error'], mis_error)
            scores['auc'] = np.append(scores['auc'], auc)
            scores['log_loss'] = np.append(scores['log_loss'], ll)

    return scores
