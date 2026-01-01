"""
Token position finding for activation extraction.

Identifies key token positions in episodes:
- first_assistant: First token of assistant response
- mid_response: Middle of assistant response
- before_tool: Last token before tool call
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TokenPosition:
    """A token position within a sequence."""

    name: str
    token_index: int
    token_str: str
    description: str


POSITION_DESCRIPTIONS = {
    "first_assistant": "First token of assistant response (before any tool content)",
    "mid_response": "Middle token of assistant response",
    "before_tool": "Last token before tool call (if present)",
    "final": "Final token of response",
}


def find_first_assistant_position(
    tokenizer,
    full_text: str,
    system_prompt: str,
    user_turns: list[str],
    backend=None,
) -> Optional[TokenPosition]:
    """
    Find the first token of the assistant's response.

    This is the critical position for ruling out syntax detection.
    At this point, the model has not yet generated any tool-related tokens.

    Args:
        tokenizer: Model tokenizer
        full_text: Full formatted prompt + response
        system_prompt: System prompt
        user_turns: User messages
        backend: Optional backend instance (to avoid reloading model)

    Returns:
        TokenPosition or None if not found
    """
    # Tokenize the full text
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Reconstruct prompt (without response)
    # Use provided backend or create one (but prefer passing backend to avoid reload)
    if backend is None:
        from ..backends import get_backend
        from ..config import get_config

        config = get_config()
        backend_class = get_backend(config.model.backend)
        backend = backend_class(
            model_id=config.model.id,
            quantization=config.model.quantization,
        )

    prompt_only = backend.format_chat(system_prompt, user_turns)
    prompt_ids = tokenizer.encode(prompt_only, add_special_tokens=False)

    # First assistant token is right after prompt
    first_assistant_idx = len(prompt_ids)

    if first_assistant_idx >= len(input_ids):
        logger.warning("First assistant position beyond sequence length")
        return None

    token_str = tokenizer.decode([input_ids[first_assistant_idx]])

    return TokenPosition(
        name="first_assistant",
        token_index=first_assistant_idx,
        token_str=token_str,
        description=POSITION_DESCRIPTIONS["first_assistant"],
    )


def find_before_tool_position(
    tokenizer,
    full_text: str,
    tool_call_pattern: str = r"<<CALL",
) -> Optional[TokenPosition]:
    """
    Find the last token before the tool call.

    Args:
        tokenizer: Model tokenizer
        full_text: Full formatted prompt + response
        tool_call_pattern: Pattern to search for

    Returns:
        TokenPosition or None if no tool call found
    """
    # Find tool call in text
    match = re.search(tool_call_pattern, full_text)
    if not match:
        return None

    # Get substring before tool call
    text_before_tool = full_text[:match.start()]

    # Tokenize
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)
    before_tool_ids = tokenizer.encode(text_before_tool, add_special_tokens=False)

    # Last token before tool
    if not before_tool_ids:
        return None

    before_tool_idx = len(before_tool_ids) - 1
    token_str = tokenizer.decode([input_ids[before_tool_idx]])

    return TokenPosition(
        name="before_tool",
        token_index=before_tool_idx,
        token_str=token_str,
        description=POSITION_DESCRIPTIONS["before_tool"],
    )


def find_mid_response_position(
    tokenizer,
    full_text: str,
    system_prompt: str,
    user_turns: list[str],
    backend=None,
) -> Optional[TokenPosition]:
    """
    Find the middle token of the assistant's response.

    Args:
        tokenizer: Model tokenizer
        full_text: Full formatted prompt + response
        system_prompt: System prompt
        user_turns: User messages
        backend: Optional backend instance (to avoid reloading model)

    Returns:
        TokenPosition or None
    """
    # Get first assistant position
    first_pos = find_first_assistant_position(
        tokenizer, full_text, system_prompt, user_turns, backend=backend
    )
    if first_pos is None:
        return None

    # Tokenize full text
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Mid is halfway between first assistant and end
    start_idx = first_pos.token_index
    end_idx = len(input_ids) - 1
    mid_idx = (start_idx + end_idx) // 2

    if mid_idx >= len(input_ids):
        return None

    token_str = tokenizer.decode([input_ids[mid_idx]])

    return TokenPosition(
        name="mid_response",
        token_index=mid_idx,
        token_str=token_str,
        description=POSITION_DESCRIPTIONS["mid_response"],
    )


def find_final_position(
    tokenizer,
    full_text: str,
) -> TokenPosition:
    """
    Find the final token of the response.

    Args:
        tokenizer: Model tokenizer
        full_text: Full formatted prompt + response

    Returns:
        TokenPosition
    """
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)
    final_idx = len(input_ids) - 1

    token_str = tokenizer.decode([input_ids[final_idx]])

    return TokenPosition(
        name="final",
        token_index=final_idx,
        token_str=token_str,
        description=POSITION_DESCRIPTIONS["final"],
    )


def find_token_position(
    tokenizer,
    full_text: str,
    position_name: str,
    system_prompt: Optional[str] = None,
    user_turns: Optional[list[str]] = None,
    backend=None,
) -> Optional[TokenPosition]:
    """
    Find a token position by name.

    Args:
        tokenizer: Model tokenizer
        full_text: Full formatted prompt + response
        position_name: Position to find ("first_assistant", "mid_response", "before_tool", "final")
        system_prompt: Required for first_assistant and mid_response
        user_turns: Required for first_assistant and mid_response
        backend: Optional backend instance (to avoid reloading model)

    Returns:
        TokenPosition or None if not found
    """
    if position_name == "first_assistant":
        if system_prompt is None or user_turns is None:
            raise ValueError("system_prompt and user_turns required for first_assistant")
        return find_first_assistant_position(tokenizer, full_text, system_prompt, user_turns, backend=backend)

    elif position_name == "mid_response":
        if system_prompt is None or user_turns is None:
            raise ValueError("system_prompt and user_turns required for mid_response")
        return find_mid_response_position(tokenizer, full_text, system_prompt, user_turns, backend=backend)

    elif position_name == "before_tool":
        return find_before_tool_position(tokenizer, full_text)

    elif position_name == "final":
        return find_final_position(tokenizer, full_text)

    else:
        raise ValueError(f"Unknown position: {position_name}")


def find_all_positions(
    tokenizer,
    full_text: str,
    system_prompt: str,
    user_turns: list[str],
    position_names: Optional[list[str]] = None,
    backend=None,
) -> dict[str, Optional[TokenPosition]]:
    """
    Find all requested positions.

    Args:
        tokenizer: Model tokenizer
        full_text: Full formatted prompt + response
        system_prompt: System prompt
        user_turns: User messages
        position_names: Positions to find (default: all)
        backend: Optional backend instance (to avoid reloading model)

    Returns:
        Dict mapping position name to TokenPosition (or None if not found)
    """
    if position_names is None:
        position_names = ["first_assistant", "mid_response", "before_tool", "final"]

    positions = {}

    for name in position_names:
        try:
            pos = find_token_position(
                tokenizer, full_text, name, system_prompt, user_turns, backend=backend
            )
            positions[name] = pos
        except Exception as e:
            logger.warning(f"Failed to find position {name}: {e}")
            positions[name] = None

    return positions


def validate_position_order(positions: dict[str, Optional[TokenPosition]]) -> bool:
    """
    Validate that positions are in expected order.

    Expected: first_assistant < mid_response < before_tool < final

    Args:
        positions: Dict of positions

    Returns:
        True if order is valid
    """
    # Get non-None positions with indices
    indexed = [
        (name, pos.token_index)
        for name, pos in positions.items()
        if pos is not None
    ]

    if len(indexed) < 2:
        return True  # Not enough to validate

    # Sort by expected order
    order = ["first_assistant", "mid_response", "before_tool", "final"]
    indexed_sorted = sorted(indexed, key=lambda x: order.index(x[0]))

    # Check indices are increasing
    for i in range(len(indexed_sorted) - 1):
        if indexed_sorted[i][1] >= indexed_sorted[i + 1][1]:
            logger.warning(
                f"Position order violation: {indexed_sorted[i][0]} ({indexed_sorted[i][1]}) "
                f">= {indexed_sorted[i + 1][0]} ({indexed_sorted[i + 1][1]})"
            )
            return False

    return True
