"""
Intervention module for causal experiments.

This module provides tools for:
- Steering vector experiments (adding/subtracting probe directions)
- Activation patching (swapping activations between episodes)
"""

from .steering import (
    SteeringExperiment,
    run_steering_experiment,
    compute_dose_response,
)
from .patching import (
    patch_activations,
    run_patching_experiment,
)

__all__ = [
    "SteeringExperiment",
    "run_steering_experiment",
    "compute_dose_response",
    "patch_activations",
    "run_patching_experiment",
]
