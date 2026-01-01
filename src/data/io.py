"""
Data I/O utilities.

Provides functions for loading and saving episodes and activations
in various formats (Parquet, JSONL, NPZ).
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .episode import Episode, EpisodeCollection
from .activation import ActivationSample, ActivationDataset

logger = logging.getLogger(__name__)


# =============================================================================
# Episode I/O
# =============================================================================


def load_episodes(
    path: Union[str, Path],
    validate: bool = True,
) -> EpisodeCollection:
    """
    Load episodes from file.

    Supports:
    - .parquet (recommended)
    - .jsonl (line-delimited JSON)
    - .json (JSON array)

    Args:
        path: Path to episode file
        validate: Whether to validate each episode against schema

    Returns:
        EpisodeCollection with loaded episodes
    """
    path = Path(path)
    logger.info(f"Loading episodes from: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Episode file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        episodes = _load_episodes_parquet(path, validate)
    elif suffix == ".jsonl":
        episodes = _load_episodes_jsonl(path, validate)
    elif suffix == ".json":
        episodes = _load_episodes_json(path, validate)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    logger.info(f"Loaded {len(episodes)} episodes")
    return EpisodeCollection(episodes=episodes)


def load_all_episodes(
    paths: Union[list[Union[str, Path]], Union[str, Path]],
    validate: bool = True,
) -> EpisodeCollection:
    """
    Load episodes from multiple files and combine them.
    
    Args:
        paths: Single path or list of paths to episode files
        validate: Whether to validate each episode against schema
        
    Returns:
        EpisodeCollection with all episodes combined
        
    Example:
        >>> # Load from multiple files
        >>> episodes = load_all_episodes([
        ...     "data/processed/new_episodes.parquet",
        ...     "data/processed/episodes.parquet"
        ... ])
        >>> 
        >>> # Or load from a single file (same as load_episodes)
        >>> episodes = load_all_episodes("data/processed/new_episodes.parquet")
    """
    if isinstance(paths, (str, Path)):
        # Single file - just use load_episodes
        return load_episodes(paths, validate=validate)
    
    # Multiple files - load and combine
    all_episodes = []
    for path in paths:
        collection = load_episodes(path, validate=validate)
        all_episodes.extend(collection.episodes)
    
    logger.info(f"Combined {len(all_episodes)} episodes from {len(paths)} files")
    return EpisodeCollection(episodes=all_episodes)


def _load_episodes_parquet(path: Path, validate: bool) -> list[Episode]:
    """Load episodes from Parquet file."""
    df = pd.read_parquet(path)
    episodes = []

    for _, row in df.iterrows():
        data = row.to_dict()
        # Handle list columns (stored as strings in some Parquet writers)
        if isinstance(data.get("user_turns"), str):
            data["user_turns"] = json.loads(data["user_turns"])
        if isinstance(data.get("tool_call_args"), str):
            data["tool_call_args"] = json.loads(data["tool_call_args"])

        if validate:
            episode = Episode(**data)
        else:
            episode = Episode.model_construct(**data)
        episodes.append(episode)

    return episodes


def _load_episodes_jsonl(path: Path, validate: bool) -> list[Episode]:
    """Load episodes from JSONL file."""
    episodes = []

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Check if this looks like legacy format
                if "scenario_id" in data or "reply" in data or data.get("category", "").endswith("_search"):
                    # Convert legacy format
                    data = _convert_legacy_episode(data)
                
                if validate:
                    episode = Episode(**data)
                else:
                    episode = Episode.model_construct(**data)
                episodes.append(episode)
            except Exception as e:
                logger.warning(f"Error on line {line_num}: {e}")
                continue

    return episodes


def _load_episodes_json(path: Path, validate: bool) -> list[Episode]:
    """Load episodes from JSON file."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "episodes" in data:
        items = data["episodes"]
    else:
        raise ValueError("JSON must be an array or contain 'episodes' key")

    episodes = []
    for item in items:
        if validate:
            episode = Episode(**item)
        else:
            episode = Episode.model_construct(**item)
        episodes.append(episode)

    return episodes


def save_episodes(
    episodes: Union[EpisodeCollection, list[Episode]],
    path: Union[str, Path],
    format: Optional[str] = None,
) -> None:
    """
    Save episodes to file.

    Args:
        episodes: Episodes to save
        path: Output path
        format: File format (inferred from extension if not provided)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(episodes, EpisodeCollection):
        episode_list = episodes.episodes
    else:
        episode_list = episodes

    suffix = format or path.suffix.lower().lstrip(".")

    if suffix == "parquet":
        _save_episodes_parquet(episode_list, path)
    elif suffix == "jsonl":
        _save_episodes_jsonl(episode_list, path)
    elif suffix == "json":
        _save_episodes_json(episode_list, path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    logger.info(f"Saved {len(episode_list)} episodes to: {path}")


def _save_episodes_parquet(episodes: list[Episode], path: Path) -> None:
    """Save episodes to Parquet file."""
    records = []
    for ep in episodes:
        record = ep.model_dump()
        # Convert lists to JSON strings for Parquet compatibility
        record["user_turns"] = json.dumps(record["user_turns"])
        if record.get("tool_call_args"):
            record["tool_call_args"] = json.dumps(record["tool_call_args"])
        records.append(record)

    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)


def _save_episodes_jsonl(episodes: list[Episode], path: Path) -> None:
    """Save episodes to JSONL file."""
    with open(path, "w") as f:
        for ep in episodes:
            line = ep.model_dump_json()
            f.write(line + "\n")


def _save_episodes_json(episodes: list[Episode], path: Path) -> None:
    """Save episodes to JSON file."""
    data = [ep.model_dump() for ep in episodes]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# =============================================================================
# Activation I/O
# =============================================================================


def load_activations(
    path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
) -> ActivationDataset:
    """
    Load activations from file.

    Supports:
    - .npz (numpy compressed archive)
    - .parquet (for metadata, with separate .npy for activations)

    Args:
        path: Path to activation file
        metadata_path: Optional separate path for metadata

    Returns:
        ActivationDataset with loaded activations
    """
    path = Path(path)
    logger.info(f"Loading activations from: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Activation file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npz":
        dataset = _load_activations_npz(path)
    elif suffix == ".parquet":
        npy_path = path.with_suffix(".npy")
        if not npy_path.exists():
            raise FileNotFoundError(f"Activation array not found: {npy_path}")
        dataset = _load_activations_parquet(path, npy_path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    logger.info(f"Loaded {len(dataset)} activation samples")
    return dataset


def _load_activations_npz(path: Path) -> ActivationDataset:
    """Load activations from NPZ file."""
    data = np.load(path, allow_pickle=True)

    # Get activation matrix
    activations = data["activations"]
    n_samples, hidden_size = activations.shape

    # Get metadata arrays
    tool_used = data["tool_used"]
    tool_used_any = data.get("tool_used_any", tool_used)  # Fallback for old files
    claims_action = data.get("claims_action", np.zeros(n_samples, dtype=bool))
    categories = data.get("categories", np.array(["unknown"] * n_samples))
    positions = data.get("positions", np.array(["unknown"] * n_samples))
    layers = data.get("layers", np.zeros(n_samples, dtype=int))
    episode_ids = data.get("episode_ids", np.array([f"ep_{i}" for i in range(n_samples)]))
    tool_types = data.get("tool_types", np.array(["unknown"] * n_samples))
    system_variants = data.get("system_variants", np.array(["unknown"] * n_samples))
    social_pressures = data.get("social_pressures", np.array(["unknown"] * n_samples))
    model_id = str(data.get("model_id", "unknown"))

    # Build samples
    samples = []
    for i in range(n_samples):
        sample = ActivationSample(
            episode_id=str(episode_ids[i]),
            position=str(positions[i]),
            layer=int(layers[i]),
            token_index=i,  # Placeholder
            tool_used=bool(tool_used[i]),
            tool_used_any=bool(tool_used_any[i]),
            claims_action=bool(claims_action[i]),
            category=str(categories[i]),
            tool_type=str(tool_types[i]),
            system_variant=str(system_variants[i]),
            social_pressure=str(social_pressures[i]),
            model_id=model_id,
        )
        samples.append(sample)

    dataset = ActivationDataset(
        samples=samples,
        model_id=model_id,
        hidden_size=hidden_size,
    )
    dataset._activations = activations

    return dataset


def _load_activations_parquet(metadata_path: Path, npy_path: Path) -> ActivationDataset:
    """Load activations from Parquet metadata + NPY array."""
    # Load metadata
    df = pd.read_parquet(metadata_path)

    # Load activations
    activations = np.load(npy_path)

    # Build samples
    samples = []
    for _, row in df.iterrows():
        sample = ActivationSample(**row.to_dict())
        samples.append(sample)

    # Get hidden size and model_id from first sample or metadata
    hidden_size = activations.shape[1]
    model_id = samples[0].model_id if samples else "unknown"

    dataset = ActivationDataset(
        samples=samples,
        model_id=model_id,
        hidden_size=hidden_size,
    )
    dataset._activations = activations

    return dataset


def save_activations(
    dataset: ActivationDataset,
    path: Union[str, Path],
    format: Optional[str] = None,
) -> None:
    """
    Save activations to file.

    Args:
        dataset: ActivationDataset to save
        path: Output path
        format: File format ("npz" or "parquet")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = format or path.suffix.lower().lstrip(".")

    if suffix == "npz":
        _save_activations_npz(dataset, path)
    elif suffix == "parquet":
        _save_activations_parquet(dataset, path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    logger.info(f"Saved {len(dataset)} activation samples to: {path}")


def _save_activations_npz(dataset: ActivationDataset, path: Path) -> None:
    """Save activations to NPZ file."""
    if dataset.activations is None:
        raise ValueError("No activations to save")

    # Prepare arrays
    data = {
        "activations": dataset.activations,
        "tool_used": np.array([s.tool_used for s in dataset.samples]),
        "tool_used_any": np.array([s.tool_used_any for s in dataset.samples]),
        "claims_action": np.array([s.claims_action for s in dataset.samples]),
        "categories": np.array([s.category for s in dataset.samples]),
        "positions": np.array([s.position for s in dataset.samples]),
        "layers": np.array([s.layer for s in dataset.samples]),
        "episode_ids": np.array([s.episode_id for s in dataset.samples]),
        "tool_types": np.array([s.tool_type for s in dataset.samples]),
        "system_variants": np.array([s.system_variant for s in dataset.samples]),
        "social_pressures": np.array([s.social_pressure for s in dataset.samples]),
        "model_id": dataset.model_id,
        "hidden_size": dataset.hidden_size,
    }

    np.savez_compressed(path, **data)


def _save_activations_parquet(dataset: ActivationDataset, path: Path) -> None:
    """Save activations to Parquet metadata + NPY array."""
    if dataset.activations is None:
        raise ValueError("No activations to save")

    # Save metadata to Parquet
    records = [s.model_dump() for s in dataset.samples]
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)

    # Save activations to NPY
    npy_path = path.with_suffix(".npy")
    np.save(npy_path, dataset.activations)


# =============================================================================
# Legacy format converters
# =============================================================================


def convert_legacy_episodes(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
) -> EpisodeCollection:
    """
    Convert legacy episode format to new schema.

    Handles field name changes and missing fields.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info(f"Converting legacy episodes: {input_path} -> {output_path}")

    # Load raw data
    with open(input_path) as f:
        if input_path.suffix == ".jsonl":
            raw_data = [json.loads(line) for line in f if line.strip()]
        else:
            raw_data = json.load(f)
            if not isinstance(raw_data, list):
                raw_data = raw_data.get("episodes", [])

    # Convert each episode
    episodes = []
    for i, raw in enumerate(raw_data):
        try:
            converted = _convert_legacy_episode(raw)
            episode = Episode(**converted)
            episodes.append(episode)
        except Exception as e:
            logger.warning(f"Failed to convert episode {i}: {e}")
            continue

    # Save
    collection = EpisodeCollection(episodes=episodes)
    save_episodes(collection, output_path)

    return collection


def _convert_legacy_episode(raw: dict) -> dict:
    """Convert a single legacy episode to new format."""
    # Map old field names to new
    field_map = {
        "scenario_id": "scenario",
        "system_prompt": "system_prompt",
        "user_turns": "user_turns",
        "reply": "assistant_reply",
        "assistant_reply": "assistant_reply",
        "tool_used": "tool_used",
        "claims_action": "claims_action",
        "claims_escalation": "claims_action",  # Legacy name
    }

    converted = {}
    for old_name, new_name in field_map.items():
        if old_name in raw:
            converted[new_name] = raw[old_name]

    # Handle tool_type
    if "tool_type" in raw:
        converted["tool_type"] = raw["tool_type"]
    elif "escalate" in str(raw.get("scenario_id", "")).lower():
        converted["tool_type"] = "escalate"
    elif "search" in str(raw.get("scenario_id", "")).lower():
        converted["tool_type"] = "search"
    else:
        converted["tool_type"] = "escalate"  # Default

    # Map legacy system_variant values (for search episodes)
    system_variant_map = {
        "baseline": "A_STRICT",
        "helpful_bias": "B_DILUTED",
        "efficiency_pressure": "C_CONFLICTING",
        "confidence_bias": "B_DILUTED",
    }
    old_system_variant = raw.get("system_variant", "")
    converted["system_variant"] = system_variant_map.get(old_system_variant, old_system_variant)

    # Map legacy social_pressure values (for search episodes)
    social_pressure_map = {
        "none": "NEUTRAL",
        "time_pressure": "STRESSED",
        "already_searched": "DEMAND",
        "expert_expectation": "VALIDATION",
        "frustration": "APPEASE",
    }
    old_social_pressure = raw.get("social_pressure", "")
    converted["social_pressure"] = social_pressure_map.get(old_social_pressure, old_social_pressure)

    # Map legacy category values
    category_map = {
        "true_search": "true_action",
        "fake_search": "fake_action",
        "honest_no_search": "honest_no_action",
        "honest_search": None,  # Need to compute based on tool_used and claims_action
    }
    old_category = raw.get("category", "")
    if old_category in category_map:
        mapped_category = category_map[old_category]
        if mapped_category is None:
            # Compute for honest_search: tool_used=True, claims_action=False -> silent_action
            tool_used = converted.get("tool_used", False)
            claims = converted.get("claims_action", False)
            if tool_used and not claims:
                converted["category"] = "silent_action"
            elif tool_used and claims:
                converted["category"] = "true_action"
            else:
                converted["category"] = "honest_no_action"
        else:
            converted["category"] = mapped_category
    else:
        # Compute category if missing or unknown
        tool_used = converted.get("tool_used", False)
        claims = converted.get("claims_action", False)
        if tool_used and claims:
            converted["category"] = "true_action"
        elif not tool_used and claims:
            converted["category"] = "fake_action"
        elif not tool_used and not claims:
            converted["category"] = "honest_no_action"
        else:
            converted["category"] = "silent_action"

    # Handle model_id
    converted["model_id"] = raw.get("model_name", raw.get("model_id", "unknown"))

    # Extract tool call info from assistant_reply if tool_used
    if converted.get("tool_used", False) and "assistant_reply" in converted:
        reply = converted["assistant_reply"]
        # Try to extract tool call
        import re
        tool_call_match = re.search(r'<<CALL\s+(\w+)\(([^)]+)\)>>', reply)
        if tool_call_match:
            tool_name = tool_call_match.group(1)
            tool_args_str = tool_call_match.group(2)
            converted["tool_call_raw"] = tool_call_match.group(0)
            # Try to parse args (simple case: query="...")
            args = {}
            if "tool_arg" in raw and raw["tool_arg"]:
                # Use tool_arg if available
                if converted["tool_type"] == "search":
                    args["query"] = raw["tool_arg"]
                else:
                    args["arg"] = raw["tool_arg"]
            converted["tool_call_args"] = args if args else None

    return converted
