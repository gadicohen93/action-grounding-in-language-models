"""
Activation patching experiments.

Implements activation patching by swapping activations between episodes.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..backends import get_backend
from ..config import get_config
from ..data import Episode
from ..labeling import detect_tool_call

logger = logging.getLogger(__name__)


@dataclass
class PatchingResult:
    """Result from an activation patching experiment."""

    source_episode_id: str
    target_episode_id: str
    layer: int
    position_index: int
    source_tool_used: bool
    target_tool_used: bool
    patched_tool_used: bool
    patched_reply: str
    effect: bool  # True if patching changed behavior


def patch_activations(
    source_episode: Episode,
    target_episode: Episode,
    layer: int,
    position_index: int,
    backend,
) -> str:
    """
    Patch activations from source episode into target episode at a specific position.

    Args:
        source_episode: Episode to take activations from
        target_episode: Episode to patch into
        layer: Layer to patch at
        position_index: Token position to patch
        backend: Model backend

    Returns:
        Patched reply text
    """
    # Format prompts
    source_text = backend.format_chat(
        source_episode.system_prompt,
        source_episode.user_turns,
        assistant_prefix=source_episode.assistant_reply,
    )

    target_prompt = backend.format_chat(
        target_episode.system_prompt,
        target_episode.user_turns,
    )

    # Get source activations
    source_hidden_states = backend.get_hidden_states(source_text, layers=[layer])
    source_activation = source_hidden_states[layer][position_index]

    # Create patching hook
    patched_activation = source_activation.to(backend.get_device())

    def patching_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Replace activation at position
        if hidden_states.dim() == 3:
            # (batch_size, seq_len, hidden_dim)
            if position_index < hidden_states.shape[1]:
                hidden_states[0, position_index, :] = patched_activation
        elif hidden_states.dim() == 2:
            # (seq_len, hidden_dim)
            if position_index < hidden_states.shape[0]:
                hidden_states[position_index, :] = patched_activation

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states

    # Register hook
    layer_module = backend.model.model.layers[layer]
    hook_handle = layer_module.register_forward_hook(patching_hook)

    try:
        # Generate with patching
        output = backend.generate(
            prompt=target_prompt,
            max_tokens=512,
            temperature=0.7,
        )
        patched_reply = output.text

    finally:
        # Remove hook
        hook_handle.remove()

    return patched_reply


def run_patching_experiment(
    source_episode: Episode,
    target_episode: Episode,
    layer: int,
    position_index: int,
    model_id: Optional[str] = None,
) -> PatchingResult:
    """
    Run activation patching experiment.

    Args:
        source_episode: Episode to take activations from
        target_episode: Episode to patch into
        layer: Layer to patch at
        position_index: Token position to patch
        model_id: Model to use (from config if None)

    Returns:
        PatchingResult
    """
    config = get_config()

    if model_id is None:
        model_id = config.model.id

    # Initialize backend
    backend_class = get_backend(config.model.backend)
    backend = backend_class(
        model_id=model_id,
        quantization=config.model.quantization,
        device_map=config.model.device_map,
        dtype=config.model.dtype,
    )

    try:
        # Patch and generate
        patched_reply = patch_activations(
            source_episode,
            target_episode,
            layer,
            position_index,
            backend,
        )

        # Detect tool usage
        patched_result = detect_tool_call(patched_reply, target_episode.tool_type)

        # Check if patching had an effect
        effect = (target_episode.tool_used != patched_result["tool_used"])

        result = PatchingResult(
            source_episode_id=source_episode.id,
            target_episode_id=target_episode.id,
            layer=layer,
            position_index=position_index,
            source_tool_used=source_episode.tool_used,
            target_tool_used=target_episode.tool_used,
            patched_tool_used=patched_result["tool_used"],
            patched_reply=patched_reply,
            effect=effect,
        )

    finally:
        backend.unload()

    return result


def run_patching_batch(
    episode_pairs: list[tuple[Episode, Episode]],
    layer: int,
    position_index: int,
    model_id: Optional[str] = None,
    verbose: bool = True,
) -> list[PatchingResult]:
    """
    Run activation patching on multiple episode pairs.

    Args:
        episode_pairs: List of (source, target) episode pairs
        layer: Layer to patch at
        position_index: Token position to patch
        model_id: Model to use
        verbose: Show progress

    Returns:
        List of PatchingResult objects
    """
    from tqdm import tqdm

    logger.info(f"Running activation patching:")
    logger.info(f"  Pairs: {len(episode_pairs)}")
    logger.info(f"  Layer: {layer}")
    logger.info(f"  Position: {position_index}")

    results = []

    iterator = tqdm(episode_pairs, desc="Patching") if verbose else episode_pairs

    for source, target in iterator:
        try:
            result = run_patching_experiment(
                source, target, layer, position_index, model_id
            )
            results.append(result)

        except Exception as e:
            logger.error(
                f"Patching failed for {source.id} → {target.id}: {e}"
            )
            continue

    return results


def create_matched_pairs(
    episodes_true: list[Episode],
    episodes_fake: list[Episode],
    n_pairs: int = 50,
) -> list[tuple[Episode, Episode]]:
    """
    Create matched pairs of true/fake episodes for patching.

    Matches episodes with same tool, variant, and pressure.

    Args:
        episodes_true: Episodes with tool_used=True
        episodes_fake: Episodes with tool_used=False (fake claims)
        n_pairs: Number of pairs to create

    Returns:
        List of (true, fake) episode pairs
    """
    pairs = []

    for fake_ep in episodes_fake:
        # Find matching true episode
        for true_ep in episodes_true:
            if (
                true_ep.tool_type == fake_ep.tool_type
                and true_ep.system_variant == fake_ep.system_variant
                and true_ep.social_pressure == fake_ep.social_pressure
                and len(pairs) < n_pairs
            ):
                pairs.append((true_ep, fake_ep))
                break

    logger.info(f"Created {len(pairs)} matched pairs")

    return pairs
