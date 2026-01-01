"""
Labeling module for episode categorization.

This module provides tools for detecting:
- Tool calls in model outputs (via regex DSL parsing)
- Action claims in model outputs (via regex or LLM judge)
"""

from .tool_detection import detect_tool_call, parse_tool_calls
from .claim_detection import detect_action_claim, detect_action_claims_batch

__all__ = [
    "detect_tool_call",
    "parse_tool_calls",
    "detect_action_claim",
    "detect_action_claims_batch",
]
