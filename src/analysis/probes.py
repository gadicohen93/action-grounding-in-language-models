"""
Probe training and evaluation.

Linear probes for detecting features in activation space.
"""

import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.exceptions import UndefinedMetricWarning

from ..data import ActivationDataset
from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class ProbeMetrics:
    """Metrics for a trained probe."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    # Cross-validation scores
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None

    # Confusion matrix
    true_positives: Optional[int] = None
    false_positives: Optional[int] = None
    true_negatives: Optional[int] = None
    false_negatives: Optional[int] = None

    def __str__(self) -> str:
        lines = [
            f"Accuracy:  {self.accuracy:.3f}",
            f"Precision: {self.precision:.3f}",
            f"Recall:    {self.recall:.3f}",
            f"F1:        {self.f1:.3f}",
        ]
        
        # ROC-AUC is undefined when only one class is present
        if not np.isnan(self.roc_auc):
            lines.append(f"ROC-AUC:   {self.roc_auc:.3f}")
        else:
            lines.append("ROC-AUC:   N/A (only one class present)")

        if self.cv_mean is not None:
            lines.append(f"CV Mean:   {self.cv_mean:.3f} ± {self.cv_std:.3f}")

        if self.true_positives is not None:
            lines.append(
                f"Confusion: TP={self.true_positives}, FP={self.false_positives}, "
                f"TN={self.true_negatives}, FN={self.false_negatives}"
            )

        return "\n".join(lines)


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    regularization: Optional[float] = None,
    n_folds: Optional[int] = None,
    random_state: int = 42,
) -> tuple[LogisticRegression, Optional[float], Optional[float]]:
    """
    Train a linear probe with cross-validation.

    Args:
        X_train: Training activations (n_samples, hidden_dim)
        y_train: Training labels (n_samples,)
        regularization: L2 regularization strength (from config if None)
        n_folds: Number of CV folds (from config if None)
        random_state: Random seed

    Returns:
        (trained_probe, cv_mean_score, cv_std_score)
    """
    config = get_config()

    if regularization is None:
        regularization = config.probe.regularization
    if n_folds is None:
        n_folds = config.probe.n_folds

    logger.info(f"Training probe with C={regularization}, n_folds={n_folds}")

    # Initialize probe
    probe = LogisticRegression(
        C=regularization,
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs',
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(probe, X_train, y_train, cv=cv, scoring='accuracy')

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    logger.info(f"CV Accuracy: {cv_mean:.3f} ± {cv_std:.3f}")

    # Train on full training set
    probe.fit(X_train, y_train)

    return probe, cv_mean, cv_std


def evaluate_probe(
    probe: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> ProbeMetrics:
    """
    Evaluate a trained probe on test data.

    Args:
        probe: Trained probe
        X_test: Test activations
        y_test: Test labels

    Returns:
        ProbeMetrics
    """
    # Predictions
    y_pred = probe.predict(X_test)
    y_proba = probe.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # ROC AUC requires both classes to be present
    unique_classes = np.unique(y_test)
    if len(unique_classes) < 2:
        roc_auc = float('nan')  # Undefined when only one class present
    else:
        # Suppress warning for single-class case (we already checked)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
            roc_auc = roc_auc_score(y_test, y_proba)

    # Confusion matrix
    tp = np.sum((y_test == 1) & (y_pred == 1))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fn = np.sum((y_test == 1) & (y_pred == 0))

    metrics = ProbeMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
    )

    logger.info(f"Test metrics:\n{metrics}")

    return metrics


def train_and_evaluate(
    dataset: ActivationDataset,
    label_type: Literal["reality", "reality_any", "narrative"] = "reality",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[LogisticRegression, ProbeMetrics, ProbeMetrics]:
    """
    Train and evaluate a probe on a dataset.

    Args:
        dataset: ActivationDataset
        label_type: "reality" (tool_used), "reality_any" (tool_used_any), or "narrative" (claims_action)
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        (probe, train_metrics, test_metrics)
    """
    logger.info(f"Training {label_type} probe")

    # Split data
    train_dataset, test_dataset = dataset.train_test_split(
        test_size=test_size,
        stratify_by="category",
        random_state=random_state,
    )

    # Get data
    X_train, y_train = train_dataset.to_sklearn_format(label_type)
    X_test, y_test = test_dataset.to_sklearn_format(label_type)

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Train probe
    probe, cv_mean, cv_std = train_probe(X_train, y_train, random_state=random_state)

    # Evaluate
    train_metrics = evaluate_probe(probe, X_train, y_train)
    train_metrics.cv_mean = cv_mean
    train_metrics.cv_std = cv_std

    test_metrics = evaluate_probe(probe, X_test, y_test)

    return probe, train_metrics, test_metrics


def save_probe(probe: LogisticRegression, path: str) -> None:
    """Save a trained probe to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(probe, f)

    logger.info(f"Saved probe to: {path}")


def load_probe(path: str) -> LogisticRegression:
    """Load a trained probe from disk."""
    with open(path, 'rb') as f:
        probe = pickle.load(f)

    logger.info(f"Loaded probe from: {path}")
    return probe


def get_probe_direction(probe: LogisticRegression) -> np.ndarray:
    """
    Get the probe direction (coefficients).

    Args:
        probe: Trained probe

    Returns:
        Probe direction as numpy array (hidden_dim,)
    """
    return probe.coef_[0]


def compare_probes(
    probe1: LogisticRegression,
    probe2: LogisticRegression,
    normalize: bool = True,
) -> dict:
    """
    Compare two probes.

    Args:
        probe1: First probe
        probe2: Second probe
        normalize: Normalize directions before comparison

    Returns:
        Dict with:
            - cosine_similarity: float
            - l2_distance: float
    """
    dir1 = get_probe_direction(probe1)
    dir2 = get_probe_direction(probe2)

    if normalize:
        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = dir2 / np.linalg.norm(dir2)

    # Cosine similarity
    cosine_sim = np.dot(dir1, dir2)

    # L2 distance
    l2_dist = np.linalg.norm(dir1 - dir2)

    return {
        "cosine_similarity": float(cosine_sim),
        "l2_distance": float(l2_dist),
    }


def analyze_probe_on_category(
    probe: LogisticRegression,
    dataset: ActivationDataset,
    category: str,
    label_type: Literal["reality", "reality_any", "narrative"] = "reality",
) -> dict:
    """
    Analyze probe performance on a specific category.

    Useful for checking if probe correctly identifies fake cases.

    Args:
        probe: Trained probe
        dataset: Full dataset
        category: Category to analyze (e.g., "fake_action")
        label_type: "reality", "reality_any", or "narrative"

    Returns:
        Dict with metrics and predictions
    """
    # Filter to category
    category_dataset = dataset.filter_by_tool(category) if category in ["escalate", "search", "sendMessage"] else None

    if category_dataset is None:
        # Filter by category field
        mask = dataset.get_category_mask(category)
        if not np.any(mask):
            logger.warning(f"No samples found for category: {category}")
            return {}

        X = dataset.activations[mask]
        if label_type == "reality":
            y = np.array([s.tool_used for s in dataset.samples])[mask]
        elif label_type == "reality_any":
            y = np.array([s.tool_used_any for s in dataset.samples])[mask]
        else:  # narrative
            y = np.array([s.claims_action for s in dataset.samples])[mask]
    else:
        X, y = category_dataset.to_sklearn_format(label_type)

    # Predictions
    y_pred = probe.predict(X)
    y_proba = probe.predict_proba(X)[:, 1]

    # Metrics
    metrics = evaluate_probe(probe, X, y)

    return {
        "n_samples": len(y),
        "metrics": metrics,
        "predictions": y_pred,
        "probabilities": y_proba,
        "true_labels": y,
    }
