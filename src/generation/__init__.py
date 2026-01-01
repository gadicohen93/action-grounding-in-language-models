"""
Episode generation module.

This module handles generating episodes for the action-grounding research,
including scenario construction, prompt formatting, and model interaction.
"""

from .prompts import (
    SystemVariant,
    SocialPressure,
    ToolType,
    Scenario,
    SCENARIOS,
    SYSTEM_PROMPTS,
    build_episode_config,
    get_all_conditions,
)
from .episodes import (
    EpisodeGenerator,
    generate_episode,
    generate_batch,
)

__all__ = [
    "SystemVariant",
    "SocialPressure",
    "ToolType",
    "Scenario",
    "SCENARIOS",
    "SYSTEM_PROMPTS",
    "build_episode_config",
    "get_all_conditions",
    "EpisodeGenerator",
    "generate_episode",
    "generate_batch",
]
