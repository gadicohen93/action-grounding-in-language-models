"""
Visualization utilities for analysis results.

Standard plotting functions for figures.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

logger = logging.getLogger(__name__)

# Set publication-quality defaults
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list[str]] = None,
    normalize: Optional[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        normalize: 'true', 'pred', 'all', or None
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        normalize=normalize,
        cmap='Blues',
        ax=ax,
    )

    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    title: str = "ROC Curve",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Area under curve
        title: Plot title
        save_path: Optional path to save

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_calibration(
    mean_predicted: np.ndarray,
    fraction_positives: np.ndarray,
    title: str = "Calibration Plot",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot calibration curve (reliability diagram).

    Args:
        mean_predicted: Mean predicted probabilities per bin
        fraction_positives: Fraction of positives per bin
        title: Plot title
        save_path: Optional path to save

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(mean_predicted, fraction_positives, 'o-', linewidth=2, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_probe_comparison(
    metrics1: dict,
    metrics2: dict,
    labels: tuple[str, str] = ("Probe 1", "Probe 2"),
    title: str = "Probe Comparison",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot comparison of two probes.

    Args:
        metrics1: Metrics dict for probe 1
        metrics2: Metrics dict for probe 2
        labels: Labels for the two probes
        title: Plot title
        save_path: Optional path to save

    Returns:
        Matplotlib figure
    """
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # Extract values
    values1 = [metrics1.get(m, (0, 0, 0))[0] for m in metric_names]
    values2 = [metrics2.get(m, (0, 0, 0))[0] for m in metric_names]

    # Bar positions
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width/2, values1, width, label=labels[0], alpha=0.8)
    ax.bar(x + width/2, values2, width, label=labels[1], alpha=0.8)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_position_accuracy(
    position_accuracies: dict[str, float],
    title: str = "Probe Accuracy by Position",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot probe accuracy at different token positions.

    Args:
        position_accuracies: Dict mapping position name to accuracy
        title: Plot title
        save_path: Optional path to save

    Returns:
        Matplotlib figure
    """
    positions = list(position_accuracies.keys())
    accuracies = list(position_accuracies.values())

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(positions, accuracies, alpha=0.8)
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Chance')
    ax.axhline(y=0.8, color='g', linestyle='--', linewidth=1, label='Target (80%)')

    ax.set_xlabel('Token Position')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_layer_analysis(
    layer_accuracies: dict[int, float],
    title: str = "Probe Accuracy by Layer",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot probe accuracy at different layers.

    Args:
        layer_accuracies: Dict mapping layer index to accuracy
        title: Plot title
        save_path: Optional path to save

    Returns:
        Matplotlib figure
    """
    layers = sorted(layer_accuracies.keys())
    accuracies = [layer_accuracies[l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(layers, accuracies, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Chance')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_transfer_matrix(
    transfer_matrix: np.ndarray,
    tool_labels: list[str],
    title: str = "Cross-Tool Transfer Matrix",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot cross-tool transfer matrix.

    Args:
        transfer_matrix: Matrix of accuracies (train_tool × test_tool)
        tool_labels: Labels for tools
        title: Plot title
        save_path: Optional path to save

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(transfer_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', rotation=270, labelpad=20)

    # Set ticks
    ax.set_xticks(np.arange(len(tool_labels)))
    ax.set_yticks(np.arange(len(tool_labels)))
    ax.set_xticklabels(tool_labels)
    ax.set_yticklabels(tool_labels)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(tool_labels)):
        for j in range(len(tool_labels)):
            text = ax.text(j, i, f'{transfer_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=14)

    ax.set_xlabel('Test Tool')
    ax.set_ylabel('Train Tool')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_fake_rate_heatmap(
    fake_rates: np.ndarray,
    variant_labels: list[str],
    pressure_labels: list[str],
    title: str = "Fake Action Rate by Condition",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot fake action rate heatmap.

    Args:
        fake_rates: Matrix of fake rates (variant × pressure)
        variant_labels: Labels for system variants
        pressure_labels: Labels for social pressures
        title: Plot title
        save_path: Optional path to save

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(fake_rates, cmap='YlOrRd', vmin=0, vmax=1.0)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fake Rate', rotation=270, labelpad=20)

    # Set ticks
    ax.set_xticks(np.arange(len(pressure_labels)))
    ax.set_yticks(np.arange(len(variant_labels)))
    ax.set_xticklabels(pressure_labels)
    ax.set_yticklabels(variant_labels)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(variant_labels)):
        for j in range(len(pressure_labels)):
            text = ax.text(j, i, f'{fake_rates[i, j]:.1%}',
                          ha="center", va="center",
                          color="white" if fake_rates[i, j] > 0.5 else "black",
                          fontsize=12)

    ax.set_xlabel('Social Pressure')
    ax.set_ylabel('System Variant')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def _save_figure(fig: plt.Figure, path: Union[str, Path]) -> None:
    """
    Save figure to disk in multiple formats.

    Args:
        fig: Matplotlib figure
        path: Path to save (without extension)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Remove extension if present
    path = path.with_suffix('')

    # Save as PDF (vector) and PNG (raster)
    fig.savefig(f"{path}.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(f"{path}.png", bbox_inches='tight', dpi=300)

    logger.info(f"Saved figure to: {path}.pdf and {path}.png")
