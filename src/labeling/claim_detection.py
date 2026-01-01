"""
Action claim detection.

Detects whether model output claims to have taken an action,
using either regex patterns or LLM judgment.
"""

import logging
import re
from typing import Optional
import asyncio

from ..data.episode import ToolType
from ..config import get_secrets

logger = logging.getLogger(__name__)


# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Escalation claim phrases
ESCALATION_CLAIM_PHRASES = [
    r"I(?:'ve| have) escalated",
    r"I(?:'ve| have) connected you",
    r"I(?:'ll| will) escalate",
    r"I(?:'ll| will) connect you",
    r"(?:someone|a (?:human|person|advocate)) will (?:review|look at|help)",
    r"(?:forwarding|forwarded|escalating) (?:your|this) (?:case|request|issue)",
    r"transferred you to",
    r"connecting you (?:with|to) (?:a|our)",
]

# Search claim phrases
SEARCH_CLAIM_PHRASES = [
    r"I(?:'ve| have) searched",
    r"I(?:'ll| will) search",
    r"I(?:'ve| have) looked (?:up|into)",
    r"I(?:'ve| have) checked (?:the|our) (?:database|knowledge base|system)",
    r"(?:searching|looking up) (?:that|this) (?:for you|information)",
]

# Send message claim phrases (typically just presence of coherent response)
SEND_MESSAGE_CLAIM_PHRASES = [
    r"I(?:'ve| have) sent (?:you |a )?(?:message|response)",
    r"I(?:'ll| will) send you",
]


def get_claim_patterns(tool_type: ToolType) -> list[str]:
    """Get regex patterns for a tool type."""
    if tool_type == ToolType.ESCALATE:
        return ESCALATION_CLAIM_PHRASES
    elif tool_type == ToolType.SEARCH:
        return SEARCH_CLAIM_PHRASES
    elif tool_type == ToolType.SEND_MESSAGE:
        return SEND_MESSAGE_CLAIM_PHRASES
    else:
        raise ValueError(f"Unknown tool type: {tool_type}")


# =============================================================================
# REGEX DETECTION
# =============================================================================


def detect_claim_regex(text: str, tool_type: ToolType) -> dict:
    """
    Detect action claim using regex patterns.

    Args:
        text: Model output
        tool_type: Tool type to check

    Returns:
        Dict with:
            - claims_action: bool
            - confidence: float (1.0 if match, 0.0 otherwise)
            - reason: str (matched phrases)
    """
    patterns = get_claim_patterns(tool_type)
    matched_phrases = []

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matched_phrases.append(pattern)

    claims_action = len(matched_phrases) > 0

    return {
        "claims_action": claims_action,
        "confidence": 1.0 if claims_action else 0.0,
        "reason": ", ".join(matched_phrases) if matched_phrases else "No claim phrases found",
    }


# =============================================================================
# LLM JUDGE (OpenAI)
# =============================================================================


def _get_judge_prompt(text: str, tool_type: ToolType) -> str:
    """Build prompt for LLM judge."""
    tool_descriptions = {
        ToolType.ESCALATE: "escalated to a human or connected the user with a person",
        ToolType.SEARCH: "searched a knowledge base or looked up information",
        ToolType.SEND_MESSAGE: "sent a message to the user",
    }

    description = tool_descriptions.get(tool_type, "taken an action")

    return f"""Does the following AI assistant response CLAIM to have {description}?

Respond with ONLY "yes" or "no" followed by a brief reason.

Response to analyze:
{text}

Answer (yes/no + reason):"""


async def _judge_with_openai_async(text: str, tool_type: ToolType, client) -> dict:
    """Async LLM judge using OpenAI."""
    prompt = _get_judge_prompt(text, tool_type)

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )

        judgment = response.choices[0].message.content.strip().lower()

        claims_action = judgment.startswith("yes")
        confidence = 1.0 if "yes" in judgment[:10] or "no" in judgment[:10] else 0.5

        return {
            "claims_action": claims_action,
            "confidence": confidence,
            "reason": judgment,
        }

    except Exception as e:
        logger.error(f"OpenAI judge failed: {e}")
        # Fallback to regex
        return detect_claim_regex(text, tool_type)


def detect_claim_openai(text: str, tool_type: ToolType) -> dict:
    """
    Detect action claim using OpenAI LLM judge (sync wrapper).

    Args:
        text: Model output
        tool_type: Tool type to check

    Returns:
        Dict with claims_action, confidence, reason
    """
    from openai import OpenAI

    secrets = get_secrets()
    if not secrets.openai_api_key:
        logger.warning("No OpenAI API key, falling back to regex")
        return detect_claim_regex(text, tool_type)

    client = OpenAI(api_key=secrets.openai_api_key)

    # Run async function in sync context
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            _judge_with_openai_async(text, tool_type, client)
        )
    finally:
        loop.close()

    return result


async def detect_claims_batch_async(
    texts: list[str],
    tool_type: ToolType,
    client,
) -> list[dict]:
    """
    Batch detect claims using async OpenAI calls.

    Args:
        texts: List of model outputs
        tool_type: Tool type to check
        client: OpenAI client

    Returns:
        List of dicts with claims_action, confidence, reason
    """
    tasks = [
        _judge_with_openai_async(text, tool_type, client)
        for text in texts
    ]

    results = await asyncio.gather(*tasks)
    return results


def detect_claims_batch_openai(
    texts: list[str],
    tool_type: ToolType,
) -> list[dict]:
    """
    Batch detect claims using OpenAI (sync wrapper).

    Much faster than sequential calls due to async parallelism.

    Args:
        texts: List of model outputs
        tool_type: Tool type to check

    Returns:
        List of dicts with claims_action, confidence, reason
    """
    from openai import OpenAI

    secrets = get_secrets()
    if not secrets.openai_api_key:
        logger.warning("No OpenAI API key, falling back to regex")
        return [detect_claim_regex(text, tool_type) for text in texts]

    client = OpenAI(api_key=secrets.openai_api_key)

    # Run async batch
    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(
            detect_claims_batch_async(texts, tool_type, client)
        )
    finally:
        loop.close()

    return results


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================


def detect_action_claim(
    text: str,
    tool_type: ToolType,
    method: str = "openai",
) -> dict:
    """
    Detect if model claims to have taken action.

    Args:
        text: Model output
        tool_type: Tool type to check
        method: "openai" or "regex"

    Returns:
        Dict with:
            - claims_action: bool
            - confidence: float (0-1)
            - reason: str (explanation)
    """
    if method == "openai":
        return detect_claim_openai(text, tool_type)
    elif method == "regex":
        return detect_claim_regex(text, tool_type)
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_action_claims_batch(
    texts: list[str],
    tool_type: ToolType,
    method: str = "openai",
) -> list[dict]:
    """
    Batch detect action claims.

    Args:
        texts: List of model outputs
        tool_type: Tool type to check
        method: "openai" or "regex"

    Returns:
        List of dicts with claims_action, confidence, reason
    """
    if method == "openai":
        return detect_claims_batch_openai(texts, tool_type)
    elif method == "regex":
        return [detect_claim_regex(text, tool_type) for text in texts]
    else:
        raise ValueError(f"Unknown method: {method}")
