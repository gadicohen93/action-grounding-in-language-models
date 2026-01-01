"""
Tool call detection via DSL parsing.

Detects tool calls in model outputs using regex patterns for the DSL:
- <<CALL escalateCase(...)>>
- <<CALL searchKnowledgeBase(...)>>
- <<CALL sendMessage(...)>>
"""

import re
from typing import Optional

from ..data.episode import ToolType


# Tool call regex patterns
TOOL_CALL_PATTERN = re.compile(
    r"<<CALL\s+(\w+)\((.*?)\)>>",
    re.DOTALL | re.IGNORECASE
)

# Tool-specific patterns
ESCALATE_PATTERN = re.compile(
    r"<<CALL\s+escalateCase\s*\(",
    re.IGNORECASE
)

SEARCH_PATTERN = re.compile(
    r"<<CALL\s+searchKnowledgeBase\s*\(",
    re.IGNORECASE
)

SEND_MESSAGE_PATTERN = re.compile(
    r"<<CALL\s+sendMessage\s*\(",
    re.IGNORECASE
)


def parse_tool_calls(text: str) -> list[dict]:
    """
    Parse all tool calls from text.

    Returns list of dicts with:
        - function: Function name
        - args_raw: Raw argument string
        - full_match: Full matched text

    Example:
        Input: "<<CALL escalateCase(message=\"help\", reason=\"urgent\")>>"
        Output: [{"function": "escalateCase", "args_raw": "message=\"help\", reason=\"urgent\"", ...}]
    """
    matches = TOOL_CALL_PATTERN.finditer(text)
    calls = []

    for match in matches:
        function_name = match.group(1)
        args_raw = match.group(2).strip()

        calls.append({
            "function": function_name,
            "args_raw": args_raw,
            "full_match": match.group(0),
            "start": match.start(),
            "end": match.end(),
        })

    return calls


def detect_tool_call(text: str, tool_type: ToolType | str) -> dict:
    """
    Detect if a specific tool was called.

    Args:
        text: Model output text
        tool_type: Expected tool type (enum or string)

    Returns:
        Dict with:
            - tool_used: bool
            - raw_call: str (if found)
            - args: dict (if parseable)
    """
    # Convert string to enum if needed
    if isinstance(tool_type, str):
        tool_type_enum = ToolType(tool_type)
    else:
        tool_type_enum = tool_type
    
    # Get pattern for tool type
    if tool_type_enum == ToolType.ESCALATE:
        pattern = ESCALATE_PATTERN
    elif tool_type_enum == ToolType.SEARCH:
        pattern = SEARCH_PATTERN
    elif tool_type_enum == ToolType.SEND_MESSAGE:
        pattern = SEND_MESSAGE_PATTERN
    else:
        raise ValueError(f"Unknown tool type: {tool_type_enum}")

    # Check for match
    match = pattern.search(text)

    if not match:
        return {
            "tool_used": False,
            "raw_call": None,
            "args": None,
        }

    # Parse full call
    all_calls = parse_tool_calls(text)
    relevant_calls = [
        c for c in all_calls
        if c["function"].lower() == tool_type_enum.value.lower().replace("_", "")
    ]

    if not relevant_calls:
        return {
            "tool_used": True,
            "raw_call": match.group(0),
            "args": None,
        }

    # Get first relevant call
    call = relevant_calls[0]

    # Try to parse arguments (basic parsing, not full eval)
    args = _parse_args_basic(call["args_raw"])

    return {
        "tool_used": True,
        "raw_call": call["full_match"],
        "args": args,
    }


def detect_any_tool_call(text: str) -> dict:
    """Detect if ANY tool was called."""
    all_calls = parse_tool_calls(text)

    if not all_calls:
        return {"tool_used_any": False, "tools_called": [], "raw_calls": []}

    return {
        "tool_used_any": True,
        "tools_called": [c["function"] for c in all_calls],
        "raw_calls": [c["full_match"] for c in all_calls],
    }


def detect_tool_call_extended(text: str, expected_tool_type: ToolType | str) -> dict:
    """Extended detection returning both expected and any tool info."""
    expected_result = detect_tool_call(text, expected_tool_type)
    any_result = detect_any_tool_call(text)

    wrong_tool_name = None
    if any_result["tool_used_any"] and not expected_result["tool_used"]:
        wrong_tool_name = any_result["tools_called"][0] if any_result["tools_called"] else "unknown"

    return {
        "tool_used": expected_result["tool_used"],
        "tool_used_any": any_result["tool_used_any"],
        "raw_call": expected_result.get("raw_call"),
        "args": expected_result.get("args"),
        "wrong_tool_name": wrong_tool_name,
        "all_tools_called": any_result["tools_called"],
    }


def _parse_args_basic(args_str: str) -> Optional[dict]:
    """
    Basic argument parser for tool calls.

    Handles simple key=value pairs with quoted strings.
    Does NOT use eval for security.

    Example:
        'message="hello", reason="urgent"' -> {"message": "hello", "reason": "urgent"}
    """
    if not args_str:
        return None

    args = {}

    # Pattern for key="value" or key='value'
    arg_pattern = re.compile(r'(\w+)\s*=\s*["\']([^"\']*)["\']')

    for match in arg_pattern.finditer(args_str):
        key = match.group(1)
        value = match.group(2)
        args[key] = value

    return args if args else None


def count_tool_calls(text: str) -> dict:
    """
    Count all tool calls by type.

    Returns dict mapping tool type to count.
    """
    calls = parse_tool_calls(text)
    counts = {}

    for call in calls:
        func = call["function"].lower()
        counts[func] = counts.get(func, 0) + 1

    return counts
