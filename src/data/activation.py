"""
Activation data schema.

Defines the structure for extracted activations with metadata.
"""

from datetime import datetime
from typing import Optional
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
import uuid


class ActivationSample(BaseModel):
    """
    A single activation sample extracted from an episode.

    Contains the activation vector and all relevant metadata for
    training probes and analysis.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Unique identifiers
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    episode_id: str = Field(description="ID of source episode")

    # Activation data (stored separately in numpy format)
    # This is the key for looking up the actual vector in the .npz file
    activation_key: Optional[str] = Field(
        default=None,
        description="Key for activation in external storage"
    )

    # Extraction metadata
    position: str = Field(description="Token position (first_assistant, mid_response, before_tool)")
    layer: int = Field(ge=0, description="Layer index (0-indexed)")
    token_index: int = Field(ge=0, description="Token index in sequence")
    token_str: Optional[str] = Field(default=None, description="String representation of token")

    # Labels (from episode)
    tool_used: bool = Field(description="Ground truth: was tool actually called?")
    tool_used_any: bool = Field(
        default=False,
        description="Was ANY tool called?"
    )
    claims_action: bool = Field(description="Does model claim to have taken action?")
    category: str = Field(description="Episode category (true_action, fake_action, etc.)")
    tool_type: str = Field(description="Type of tool (escalate, search, sendMessage)")

    # Additional metadata
    system_variant: str = Field(description="System prompt variant")
    social_pressure: str = Field(description="Social pressure condition")
    model_id: str = Field(description="Model used")
    extraction_timestamp: datetime = Field(default_factory=datetime.now)

    def to_probe_labels(self) -> dict[str, int]:
        """Get binary labels for probe training."""
        return {
            "reality": int(self.tool_used),
            "narrative": int(self.claims_action),
        }


class ActivationDataset(BaseModel):
    """
    A dataset of activation samples with the actual activation vectors.

    Separates metadata (validated via Pydantic) from heavy numpy data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Metadata
    samples: list[ActivationSample] = Field(default_factory=list)
    description: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)
    model_id: str = Field(description="Model used for extraction")
    hidden_size: int = Field(ge=1, description="Hidden dimension size")

    # Activation vectors stored as numpy array
    # Shape: (n_samples, hidden_size)
    # This is not validated by Pydantic but stored alongside
    _activations: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, idx) -> ActivationSample:
        return self.samples[idx]

    @property
    def activations(self) -> Optional[np.ndarray]:
        """Get activation matrix."""
        return self._activations

    @activations.setter
    def activations(self, value: np.ndarray):
        """Set activation matrix with validation."""
        if value is not None:
            assert value.ndim == 2, "Activations must be 2D (n_samples, hidden_size)"
            assert value.shape[0] == len(self.samples), (
                f"Activation count ({value.shape[0]}) must match sample count ({len(self.samples)})"
            )
            assert value.shape[1] == self.hidden_size, (
                f"Hidden size ({value.shape[1]}) must match config ({self.hidden_size})"
            )
        self._activations = value

    def get_activation(self, idx: int) -> np.ndarray:
        """Get activation vector for a sample by index."""
        if self._activations is None:
            raise ValueError("Activations not loaded")
        return self._activations[idx]

    def to_sklearn_format(
        self,
        label_type: str = "reality"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert to sklearn-compatible format.

        Args:
            label_type: "reality" (tool_used), "reality_any" (tool_used_any), or "narrative" (claims_action)

        Returns:
            X: Activation matrix (n_samples, hidden_size)
            y: Label array (n_samples,)
        """
        if self._activations is None:
            raise ValueError("Activations not loaded")

        X = self._activations

        if label_type == "reality":
            y = np.array([s.tool_used for s in self.samples], dtype=np.int32)
        elif label_type == "reality_any":
            y = np.array([s.tool_used_any for s in self.samples], dtype=np.int32)
        elif label_type == "narrative":
            y = np.array([s.claims_action for s in self.samples], dtype=np.int32)
        else:
            raise ValueError(f"Unknown label type: {label_type}. Must be 'reality', 'reality_any', or 'narrative'")

        return X, y

    def get_category_mask(self, category: str) -> np.ndarray:
        """Get boolean mask for a specific category."""
        return np.array([s.category == category for s in self.samples])

    def filter_by_position(self, position: str) -> "ActivationDataset":
        """Filter samples by token position."""
        indices = [i for i, s in enumerate(self.samples) if s.position == position]
        return self._filter_by_indices(indices)

    def filter_by_layer(self, layer: int) -> "ActivationDataset":
        """Filter samples by layer."""
        indices = [i for i, s in enumerate(self.samples) if s.layer == layer]
        return self._filter_by_indices(indices)

    def filter_by_tool(self, tool_type: str) -> "ActivationDataset":
        """Filter samples by tool type."""
        indices = [i for i, s in enumerate(self.samples) if s.tool_type == tool_type]
        return self._filter_by_indices(indices)

    def _filter_by_indices(self, indices: list[int]) -> "ActivationDataset":
        """Create a filtered dataset by indices."""
        filtered_samples = [self.samples[i] for i in indices]
        filtered_activations = self._activations[indices] if self._activations is not None else None

        dataset = ActivationDataset(
            samples=filtered_samples,
            description=f"Filtered from {len(self.samples)} to {len(filtered_samples)}",
            model_id=self.model_id,
            hidden_size=self.hidden_size,
        )
        dataset._activations = filtered_activations
        return dataset

    def train_test_split(
        self,
        test_size: float = 0.2,
        stratify_by: str = "category",
        random_state: int = 42,
    ) -> tuple["ActivationDataset", "ActivationDataset"]:
        """
        Split dataset into train and test sets.

        Args:
            test_size: Fraction for test set
            stratify_by: Field to stratify by ("category", "tool_type", etc.)
            random_state: Random seed

        Returns:
            (train_dataset, test_dataset)
        """
        from sklearn.model_selection import train_test_split

        # Get stratification labels
        stratify_labels = [getattr(s, stratify_by) for s in self.samples]

        # Split indices
        indices = list(range(len(self.samples)))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=random_state,
        )

        train_dataset = self._filter_by_indices(train_idx)
        test_dataset = self._filter_by_indices(test_idx)

        return train_dataset, test_dataset

    def summary(self) -> dict:
        """Get summary statistics."""
        from collections import Counter

        categories = Counter(s.category for s in self.samples)
        tools = Counter(s.tool_type for s in self.samples)
        positions = Counter(s.position for s in self.samples)
        layers = Counter(s.layer for s in self.samples)

        return {
            "n_samples": len(self.samples),
            "hidden_size": self.hidden_size,
            "has_activations": self._activations is not None,
            "categories": dict(categories),
            "tools": dict(tools),
            "positions": dict(positions),
            "layers": dict(layers),
        }
