"""
Data schemas and I/O utilities for the action-grounding research project.

This module provides validated data structures for episodes and activations,
plus utilities for loading and saving data in various formats.
"""

from .episode import Episode, EpisodeCategory, ToolType
from .activation import ActivationSample, ActivationDataset
from .io import (
    load_episodes,
    save_episodes,
    load_activations,
    save_activations,
)

__all__ = [
    "Episode",
    "EpisodeCategory",
    "ToolType",
    "ActivationSample",
    "ActivationDataset",
    "load_episodes",
    "save_episodes",
    "load_activations",
    "save_activations",
]
