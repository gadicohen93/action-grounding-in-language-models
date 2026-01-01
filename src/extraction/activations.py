"""
Activation extraction from episodes.

Extracts hidden state activations at specific token positions and layers
for probe training and analysis.
"""

import logging
import numpy as np
from typing import Optional
from tqdm import tqdm

from ..backends import get_backend
from ..config import get_config
from ..data import Episode, ActivationSample, ActivationDataset
from .positions import find_token_position, find_all_positions

logger = logging.getLogger(__name__)


class ActivationExtractor:
    """
    Extractor for model activations from episodes.

    Uses backend abstraction to extract hidden states at specific
    token positions and layers.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        backend_type: Optional[str] = None,
        quantization: Optional[str] = None,
    ):
        """
        Initialize the activation extractor.

        Args:
            model_id: Model to use (from config if None)
            backend_type: Backend type (from config if None)
            quantization: Quantization level (from config if None)
        """
        config = get_config()

        self.model_id = model_id or config.model.id
        self.backend_type = backend_type or config.model.backend
        self.quantization = quantization or config.model.quantization

        # Initialize backend (lazy loaded)
        backend_class = get_backend(self.backend_type)
        self.backend = backend_class(
            model_id=self.model_id,
            quantization=self.quantization,
            device_map=config.model.device_map,
            dtype=config.model.dtype,
        )

        # Get model dimensions
        self.hidden_size = self.backend.get_hidden_size()
        self.n_layers = self.backend.get_num_layers()

        logger.info(f"Initialized ActivationExtractor:")
        logger.info(f"  Model: {self.model_id}")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Layers: {self.n_layers}")

    def extract(
        self,
        episode: Episode,
        positions: list[str],
        layers: list[int],
    ) -> list[ActivationSample]:
        """
        Extract activations from a single episode.

        Args:
            episode: Episode to extract from
            positions: Position names to extract at
            layers: Layer indices to extract from

        Returns:
            List of ActivationSample objects (one per position × layer combination)
        """
        # Build full text (prompt + response)
        full_text = self.backend.format_chat(
            episode.system_prompt,
            episode.user_turns,
            assistant_prefix=episode.assistant_reply,
        )

        # Find token positions (pass backend to avoid reloading model)
        position_objs = find_all_positions(
            self.backend.tokenizer,
            full_text,
            episode.system_prompt,
            episode.user_turns,
            position_names=positions,
            backend=self.backend,
        )

        # Extract hidden states for all layers
        hidden_states = self.backend.get_hidden_states(full_text, layers=layers)

        # Build samples
        samples = []

        for position_name in positions:
            pos = position_objs.get(position_name)
            if pos is None:
                logger.warning(
                    f"Position {position_name} not found in episode {episode.id}"
                )
                continue

            for layer in layers:
                if layer not in hidden_states:
                    logger.warning(f"Layer {layer} not in hidden states")
                    continue

                # Get activation at this position
                layer_activations = hidden_states[layer]  # Shape: (seq_len, hidden_size)
                if pos.token_index >= layer_activations.shape[0]:
                    logger.warning(
                        f"Token index {pos.token_index} out of bounds "
                        f"(seq_len={layer_activations.shape[0]})"
                    )
                    continue

                activation_vector = layer_activations[pos.token_index].cpu().numpy()

                # Create sample
                sample = ActivationSample(
                    episode_id=episode.id,
                    position=position_name,
                    layer=layer,
                    token_index=pos.token_index,
                    token_str=pos.token_str,
                    tool_used=episode.tool_used,
                    tool_used_any=episode.tool_used_any,
                    claims_action=episode.claims_action,
                    category=episode.category.value if hasattr(episode.category, 'value') else episode.category,
                    tool_type=episode.tool_type.value if hasattr(episode.tool_type, 'value') else episode.tool_type,
                    system_variant=episode.system_variant.value if hasattr(episode.system_variant, 'value') else episode.system_variant,
                    social_pressure=episode.social_pressure.value if hasattr(episode.social_pressure, 'value') else episode.social_pressure,
                    model_id=self.model_id,
                )

                # Store activation vector separately (will be collected into matrix)
                sample._activation = activation_vector
                samples.append(sample)

        return samples

    def extract_batch(
        self,
        episodes: list[Episode],
        positions: Optional[list[str]] = None,
        layers: Optional[list[int]] = None,
        verbose: bool = True,
    ) -> ActivationDataset:
        """
        Extract activations from a batch of episodes.

        Args:
            episodes: Episodes to extract from
            positions: Position names (from config if None)
            layers: Layer indices (from config if None)
            verbose: Show progress bar

        Returns:
            ActivationDataset with all samples
        """
        import time
        from datetime import datetime

        config = get_config()

        if positions is None:
            positions = config.extraction.positions
        if layers is None:
            layers = config.extraction.layers

        start_time = time.time()

        if verbose:
            logger.info("=" * 60)
            logger.info("ACTIVATION EXTRACTION")
            logger.info("=" * 60)
            logger.info(f"  Started at: {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"  Episodes: {len(episodes)}")
            logger.info(f"  Positions: {positions}")
            logger.info(f"  Layers: {layers}")
            logger.info(f"  Expected samples: {len(episodes)} × {len(positions)} × {len(layers)} = {len(episodes) * len(positions) * len(layers)}")

            # Log GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"  GPU Memory: {allocated:.1f}GB allocated / {reserved:.1f}GB reserved")
            except Exception:
                pass

        all_samples = []
        all_activations = []

        iterator = tqdm(episodes, desc="Extracting activations", unit="episode") if verbose else episodes

        for i, episode in enumerate(iterator):
            try:
                samples = self.extract(episode, positions, layers)

                for sample in samples:
                    # Extract the activation vector and remove from sample
                    activation = sample._activation
                    delattr(sample, '_activation')

                    all_samples.append(sample)
                    all_activations.append(activation)

                # Log progress every 50 episodes
                if verbose and (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining_episodes = len(episodes) - (i + 1)
                    eta = remaining_episodes / rate if rate > 0 else 0
                    logger.info(f"  Progress: {i+1}/{len(episodes)} episodes | "
                              f"Rate: {rate:.1f} eps/s | "
                              f"ETA: {eta/60:.1f} min")

            except Exception as e:
                logger.error(f"Failed to extract from episode {episode.id}: {e}")
                continue

        # Stack activations into matrix
        if all_activations:
            activation_matrix = np.stack(all_activations)
        else:
            activation_matrix = np.zeros((0, self.hidden_size))

        # Build dataset
        dataset = ActivationDataset(
            samples=all_samples,
            model_id=self.model_id,
            hidden_size=self.hidden_size,
            description=f"Extracted from {len(episodes)} episodes",
        )
        dataset.activations = activation_matrix

        total_time = time.time() - start_time

        if verbose:
            logger.info("=" * 60)
            logger.info("EXTRACTION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"  Extracted {len(all_samples)} activation samples")
            logger.info(f"  Shape: {activation_matrix.shape}")
            logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            logger.info(f"  Speed: {len(episodes)/total_time:.2f} episodes/sec")
            logger.info(f"  Average: {total_time/len(episodes):.2f}s per episode")

        return dataset

    def unload(self):
        """Unload model from memory."""
        self.backend.unload()


# =============================================================================
# Convenience Functions
# =============================================================================


def extract_activations(
    episode: Episode,
    positions: Optional[list[str]] = None,
    layers: Optional[list[int]] = None,
    model_id: Optional[str] = None,
    backend_type: Optional[str] = None,
) -> list[ActivationSample]:
    """
    Extract activations from a single episode (convenience function).

    Args:
        episode: Episode to extract from
        positions: Position names (from config if None)
        layers: Layer indices (from config if None)
        model_id: Model to use (from config if None)
        backend_type: Backend to use ("pytorch" or "vllm"). 
                     Defaults to config. Use "pytorch" for extraction
                     (vLLM doesn't support activation extraction).

    Returns:
        List of ActivationSample objects
    """
    config = get_config()

    if positions is None:
        positions = config.extraction.positions
    if layers is None:
        layers = config.extraction.layers

    # Default to pytorch for extraction (vLLM doesn't support it)
    if backend_type is None:
        backend_type = config.model.backend
        if backend_type == "vllm":
            logger.warning(
                "Config specifies vLLM backend, but vLLM doesn't support activation extraction. "
                "Switching to PyTorch backend for extraction."
            )
            backend_type = "pytorch"

    extractor = ActivationExtractor(
        model_id=model_id,
        backend_type=backend_type,
    )

    try:
        samples = extractor.extract(episode, positions, layers)
    finally:
        extractor.unload()

    return samples


def extract_activations_batch(
    episodes: list[Episode],
    positions: Optional[list[str]] = None,
    layers: Optional[list[int]] = None,
    model_id: Optional[str] = None,
    backend_type: Optional[str] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> ActivationDataset:
    """
    Extract activations from a batch of episodes (convenience function).

    **Workflow tip:** You can generate episodes with vLLM (fast) and extract
    activations later with PyTorch. Just pass `backend_type="pytorch"` here.

    Args:
        episodes: Episodes to extract from
        positions: Position names (from config if None)
        layers: Layer indices (from config if None)
        model_id: Model to use (from config if None)
        backend_type: Backend to use ("pytorch" or "vllm").
                     Defaults to config. Use "pytorch" for extraction
                     (vLLM doesn't support activation extraction).
        save_path: Optional path to save dataset
        verbose: Show progress bar

    Returns:
        ActivationDataset
    """
    config = get_config()

    # Default to pytorch for extraction (vLLM doesn't support it)
    if backend_type is None:
        backend_type = config.model.backend
        if backend_type == "vllm":
            logger.warning(
                "Config specifies vLLM backend, but vLLM doesn't support activation extraction. "
                "Switching to PyTorch backend for extraction."
            )
            backend_type = "pytorch"

    extractor = ActivationExtractor(
        model_id=model_id,
        backend_type=backend_type,
    )

    try:
        dataset = extractor.extract_batch(episodes, positions, layers, verbose)
    finally:
        extractor.unload()

    # Save if requested
    if save_path:
        from ..data.io import save_activations
        save_activations(dataset, save_path)
        logger.info(f"Saved activations to: {save_path}")

    return dataset
