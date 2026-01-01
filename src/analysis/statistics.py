"""
Statistical analysis utilities.

Bootstrap confidence intervals, significance tests, etc.
"""

import logging
import warnings
from typing import Callable, Optional

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils import resample

from ..config import get_config

logger = logging.getLogger(__name__)


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable,
    n_bootstrap: Optional[int] = None,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Data to bootstrap (1D array)
        statistic: Function to compute statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples (from config if None)
        confidence_level: Confidence level (default: 0.95)
        random_state: Random seed

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    config = get_config()

    if n_bootstrap is None:
        n_bootstrap = config.probe.bootstrap_samples

    # Compute point estimate
    point_estimate = statistic(data)

    # Bootstrap
    np.random.seed(random_state)
    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        sample = resample(data, random_state=random_state + _)
        bootstrap_estimates.append(statistic(sample))

    bootstrap_estimates = np.array(bootstrap_estimates)

    # Confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)

    return point_estimate, lower_bound, upper_bound


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_bootstrap: Optional[int] = None,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Compute bootstrap confidence intervals for multiple metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for ROC-AUC)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_state: Random seed

    Returns:
        Dict with metrics and CIs
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    config = get_config()
    if n_bootstrap is None:
        n_bootstrap = config.probe.bootstrap_samples

    np.random.seed(random_state)

    # Store bootstrap estimates
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []

    for i in range(n_bootstrap):
        # Resample
        indices = resample(
            np.arange(len(y_true)),
            random_state=random_state + i,
        )

        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Compute metrics
        accuracies.append(accuracy_score(y_true_boot, y_pred_boot))
        precisions.append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
        recalls.append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
        f1s.append(f1_score(y_true_boot, y_pred_boot, zero_division=0))

        if y_proba is not None:
            y_proba_boot = y_proba[indices]
            unique_classes = np.unique(y_true_boot)
            if len(unique_classes) >= 2:
                # Suppress warning for single-class case
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                    try:
                        aucs.append(roc_auc_score(y_true_boot, y_proba_boot))
                    except ValueError:
                        # Handle case where all samples are one class
                        pass

    # Compute CIs
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    def get_ci(values):
        return (
            np.mean(values),
            np.percentile(values, lower_percentile),
            np.percentile(values, upper_percentile),
        )

    results = {
        "accuracy": get_ci(accuracies),
        "precision": get_ci(precisions),
        "recall": get_ci(recalls),
        "f1": get_ci(f1s),
    }

    if aucs:
        results["roc_auc"] = get_ci(aucs)

    return results


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: 'true', 'pred', 'all', or None

    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred, normalize=normalize)


def compute_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC-AUC and curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities

    Returns:
        (auc, fpr, tpr, thresholds)
    """
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        auc = float('nan')
        # Return dummy curve values
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        thresholds = np.array([1.0, 0.0])
    else:
        # Suppress warning for single-class case
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
            auc = roc_auc_score(y_true, y_proba)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    return auc, fpr, tpr, thresholds


def compute_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve (reliability diagram).

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins

    Returns:
        (mean_predicted_probs, fraction_positives)
    """
    from sklearn.calibration import calibration_curve

    fraction_positives, mean_predicted = calibration_curve(
        y_true,
        y_proba,
        n_bins=n_bins,
        strategy='uniform',
    )

    return mean_predicted, fraction_positives


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    model,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Permutation test to assess if model performance is better than chance.

    Args:
        X: Features
        y: Labels
        model: Trained model
        n_permutations: Number of permutations
        random_state: Random seed

    Returns:
        (p_value, observed_score)
    """
    from sklearn.model_selection import cross_val_score

    # Observed score
    observed_score = cross_val_score(model, X, y, cv=5).mean()

    # Permutation scores
    np.random.seed(random_state)
    permutation_scores = []

    for i in range(n_permutations):
        y_perm = np.random.permutation(y)
        score = cross_val_score(model, X, y_perm, cv=5).mean()
        permutation_scores.append(score)

    # P-value
    p_value = np.mean(np.array(permutation_scores) >= observed_score)

    return p_value, observed_score


def chi_squared_test(
    contingency_table: np.ndarray,
) -> tuple[float, float]:
    """
    Chi-squared test of independence.

    Args:
        contingency_table: Contingency table (2D array)

    Returns:
        (chi2_statistic, p_value)
    """
    from scipy.stats import chi2_contingency

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    return chi2, p_value


def mcnemar_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
) -> tuple[float, float]:
    """
    McNemar's test to compare two classifiers.

    Args:
        y_true: True labels
        y_pred1: Predictions from classifier 1
        y_pred2: Predictions from classifier 2

    Returns:
        (statistic, p_value)
    """
    from statsmodels.stats.contingency_tables import mcnemar

    # Build contingency table
    # Rows: classifier 1 correct/incorrect
    # Cols: classifier 2 correct/incorrect
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    table = np.array([
        [np.sum(correct1 & correct2), np.sum(correct1 & ~correct2)],
        [np.sum(~correct1 & correct2), np.sum(~correct1 & ~correct2)],
    ])

    result = mcnemar(table, exact=True)

    return result.statistic, result.pvalue


def compute_effect_size(
    group1: np.ndarray,
    group2: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        group1: First group
        group2: Second group

    Returns:
        Cohen's d
    """
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)

    n1 = len(group1)
    n2 = len(group2)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    # Cohen's d
    d = (mean1 - mean2) / pooled_std

    return d
