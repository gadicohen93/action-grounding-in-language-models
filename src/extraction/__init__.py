"""
Activation extraction module.

This module handles extracting hidden state activations from episodes
for probe training and analysis.
"""

from .activations import (
    ActivationExtractor,
    extract_activations,
    extract_activations_batch,
)
from .positions import (
    find_token_position,
    TokenPosition,
    POSITION_DESCRIPTIONS,
)

__all__ = [
    "ActivationExtractor",
    "extract_activations",
    "extract_activations_batch",
    "find_token_position",
    "TokenPosition",
    "POSITION_DESCRIPTIONS",
]
