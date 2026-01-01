"""
Analysis module for probes, statistics, and visualization.

This module provides tools for:
- Training and evaluating linear probes
- Statistical analysis (bootstrap CIs, significance tests)
- Visualization of results
"""

from .probes import (
    train_probe,
    evaluate_probe,
    ProbeMetrics,
    compare_probes,
)
from .statistics import (
    bootstrap_ci,
    bootstrap_metrics,
    compute_confusion_matrix,
    compute_roc_auc,
)
from .visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_calibration,
    plot_probe_comparison,
)

__all__ = [
    "train_probe",
    "evaluate_probe",
    "ProbeMetrics",
    "compare_probes",
    "bootstrap_ci",
    "bootstrap_metrics",
    "compute_confusion_matrix",
    "compute_roc_auc",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_calibration",
    "plot_probe_comparison",
]
