import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    log_loss
)
import time
from datetime import datetime
from model.sp_hnb import SemiparametricHNB
from model.hnb import Hidden_NB
from model.kd_nb import KDENaiveBayes

np.random.seed(42)

def generate_correlated_data(n, p, rho, dist_type):
    """
    Generate X, y with feature correlation and different marginal shapes
    
    Parameters:
    -----------
    n : int
        Total sample size
    p : int
        Number of features
    rho : float
        Correlation coefficient
    dist_type : str
        One of "normal", "skewed", or "multimodal"
    """
    # Correlation matrix with uniform correlation
    Sigma = np.full((p, p), rho)
    np.fill_diagonal(Sigma, 1)

    X_list = []
    y_list = []

    for c in [0, 1]:
        mean = np.zeros(p) + c * 1.5

        if dist_type == "normal":
            # Standard multivariate normal
            Xc = multivariate_normal.rvs(mean=mean, cov=Sigma, size=n//2)

        elif dist_type == "skewed":
            # Skewed distribution
            Z = multivariate_normal.rvs(mean=np.zeros(p), cov=Sigma, size=n//2)
            Xc = np.sign(Z) * np.abs(Z)**(1.5) + mean

        elif dist_type == "multimodal":
            # Mixture of two Gaussians
            mix = np.random.binomial(1, 0.5, size=n//2)
            X1 = multivariate_normal.rvs(mean=mean - 1.0, cov=Sigma, size=n//2)
            X2 = multivariate_normal.rvs(mean=mean + 1.0, cov=Sigma, size=n//2)
            Xc = mix[:, None] * X1 + (1 - mix)[:, None] * X2

        X_list.append(Xc)
        y_list.append(np.full(n//2, c))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return X, y


def compute_metrics(y_true, y_pred, y_proba):
    """
    Returns dict with:
    - Accuracy 
    - Misclassification Error
    - Precision
    - Recall
    - F1-Score 
    - AUC (Area Under ROC Curve)
    - Log-Loss 
    """
    metrics = {}
    
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        metrics['misclassification_error'] = 1 - metrics['accuracy']
        
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        if y_proba is not None and len(np.unique(y_true)) == 2:
            metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics['auc'] = np.nan
        
        if y_proba is not None:
            metrics['log_loss'] = log_loss(y_true, y_proba)
        else:
            metrics['log_loss'] = np.nan
            
    except Exception as e:
        print(f"    [WARNING] Metric computation failed: {e}")
        for key in ['accuracy', 'misclassification_error', 'precision', 
                    'recall', 'f1_score', 'auc', 'log_loss']:
            metrics[key] = np.nan
    
    return metrics


def run_simulation():
    """
    - Sample sizes: 100, 500, 1000, 5000
    - Number of features: 5, 10, 20
    - Correlation: 0.1, 0.5, 0.9
    - Distributions: normal, skewed, multimodal
    
    Performance metrics averaged over 100 replications per scenario
    """
    sample_sizes = [100, 500, 1000, 5000]
    feature_sizes = [5, 10, 20]
    rhos = [0.1, 0.5, 0.9]
    dists = ["normal", "skewed", "multimodal"]
    
    n_replications = 100 

    results = []

    total_scenarios = len(sample_sizes) * len(feature_sizes) * len(rhos) * len(dists)
    total_runs = total_scenarios * n_replications
    run_count = 0
    
    start_time = time.time()

    for n in sample_sizes:
        for p in feature_sizes:
            for rho in rhos:
                for dist in dists:
                    scenario_start = time.time()
                    
                    print(f"\n[Scenario] n={n}, p={p}, rho={rho}, dist={dist}")
                    
                    for rep in range(n_replications):
                        run_count += 1
                        
                        # Data
                        X, y = generate_correlated_data(n, p, rho, dist)
                        
                        # 70-30 train-test split (stratified)
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.3, stratify=y, random_state=rep
                        )

                        result_row = {
                            "n": n,
                            "p": p,
                            "rho": rho,
                            "dist": dist,
                            "rep": rep
                        }

                        # --- Semiparametric Hidden Naive Bayes ---
                        try:
                            shnb = SemiparametricHNB(k_neighbors=5, ucv_grid_points=20)
                            shnb.fit(X_train, y_train)
                            y_pred_shnb = shnb.predict(X_test)
                            y_proba_shnb = shnb.predict_proba(X_test)
                            
                            metrics_shnb = compute_metrics(y_test, y_pred_shnb, y_proba_shnb)
                            for metric, value in metrics_shnb.items():
                                result_row[f'SHNB_{metric}'] = value
                                
                        except Exception as e:
                            print(f"    [ERROR] SHNB rep {rep}: {e}")
                            for metric in ['accuracy', 'misclassification_error', 'precision', 
                                         'recall', 'f1_score', 'auc', 'log_loss']:
                                result_row[f'SHNB_{metric}'] = np.nan
                        
                        # --- Hidden Naive Bayes ---
                        try:
                            hnb = Hidden_NB(n_bins=10)
                            hnb.fit(X_train, y_train)
                            y_pred_hnb = hnb.predict(X_test)
                            y_proba_hnb = hnb.predict_proba(X_test)
                            
                            metrics_hnb = compute_metrics(y_test, y_pred_hnb, y_proba_hnb)
                            for metric, value in metrics_hnb.items():
                                result_row[f'HNB_{metric}'] = value
                                
                        except Exception as e:
                            print(f"    [ERROR] HNB rep {rep}: {e}")
                            for metric in ['accuracy', 'misclassification_error', 'precision', 
                                         'recall', 'f1_score', 'auc', 'log_loss']:
                                result_row[f'HNB_{metric}'] = np.nan

                        # --- Gaussian Naive Bayes ---
                        try:
                            gnb = GaussianNB()
                            gnb.fit(X_train, y_train)
                            y_pred_gnb = gnb.predict(X_test)
                            y_proba_gnb = gnb.predict_proba(X_test)
                            
                            metrics_gnb = compute_metrics(y_test, y_pred_gnb, y_proba_gnb)
                            for metric, value in metrics_gnb.items():
                                result_row[f'GNB_{metric}'] = value
                                
                        except Exception as e:
                            print(f"    [ERROR] GNB rep {rep}: {e}")
                            for metric in ['accuracy', 'misclassification_error', 'precision', 
                                         'recall', 'f1_score', 'auc', 'log_loss']:
                                result_row[f'GNB_{metric}'] = np.nan

                        # --- Kernel Density Naive Bayes ---
                        try:
                            kde_nb = KDENaiveBayes()
                            kde_nb.fit(X_train, y_train)
                            y_pred_kde = kde_nb.predict(X_test)
                            # Check if predict_proba exists
                            if hasattr(kde_nb, 'predict_proba'):
                                y_proba_kde = kde_nb.predict_proba(X_test)
                            else:
                                y_proba_kde = None
                            
                            metrics_kde = compute_metrics(y_test, y_pred_kde, y_proba_kde)
                            for metric, value in metrics_kde.items():
                                result_row[f'KDE_NB_{metric}'] = value
                                
                        except Exception as e:
                            print(f"    [ERROR] KDE_NB rep {rep}: {e}")
                            for metric in ['accuracy', 'misclassification_error', 'precision', 
                                         'recall', 'f1_score', 'auc', 'log_loss']:
                                result_row[f'KDE_NB_{metric}'] = np.nan

                        results.append(result_row)

                        # Update after every 10 replications
                        if (rep + 1) % 10 == 0:
                            elapsed = time.time() - start_time
                            avg_time_per_run = elapsed / run_count
                            remaining_runs = total_runs - run_count
                            eta_seconds = avg_time_per_run * remaining_runs
                            eta_minutes = eta_seconds / 60
                            
                            print(f"  Progress: {rep + 1}/{n_replications} reps "
                                  f"({run_count}/{total_runs} total runs, "
                                  f"ETA: {eta_minutes:.1f} min)")
                    
                    scenario_time = time.time() - scenario_start
                    print(f"  Scenario completed in {scenario_time/60:.2f} minutes")

    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return pd.DataFrame(results)


def summarize_results(df_results):
    """
    Summarize simulation results by averaging over replications
    Computes mean and std for all metrics across replications
    """
    # Get all metric columns
    metric_cols = [col for col in df_results.columns 
                   if any(model in col for model in ['SHNB', 'GNB', 'KDE_NB', 'HNB'])]
    
    # Group by scenario and compute statistics
    agg_dict = {col: ['mean', 'std', 'count'] for col in metric_cols}
    
    summary = df_results.groupby(['n', 'p', 'rho', 'dist']).agg(agg_dict).reset_index()
    
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                       for col in summary.columns.values]
    
    return summary


if __name__ == "__main__":
    df_results = run_simulation()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    detailed_file = f"simulation_results_detailed_{timestamp}.csv"
    df_results.to_csv(detailed_file, index=False)
    print(f"\n✓ Detailed results saved to '{detailed_file}'")
    
    # Save summary
    df_summary = summarize_results(df_results)
    summary_file = f"simulation_results_summary_{timestamp}.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"✓ Summary results saved to '{summary_file}'")
    
    # Print quick summary statistics for all metrics
    metrics = ['accuracy', 'misclassification_error', 'precision', 'recall', 
               'f1_score', 'auc', 'log_loss']
    models = ['SHNB', 'GNB', 'KDE_NB', 'HNB']
    
    for metric in metrics:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        for model in models:
            col_name = f'{model}_{metric}'
            if col_name in df_results.columns:
                mean_val = df_results[col_name].mean()
                std_val = df_results[col_name].std()
                print(f"  {model:12s}: {mean_val:.4f} ± {std_val:.4f}")