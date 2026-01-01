"""
Abstract base class for model backends.

Defines the interface that all backends must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor


@dataclass
class GenerationOutput:
    """Output from text generation."""

    text: str
    input_ids: Optional[Tensor] = None
    output_ids: Optional[Tensor] = None
    num_tokens_generated: int = 0


class ModelBackend(ABC):
    """
    Abstract base class for language model backends.

    Provides a unified interface for model loading, generation,
    and activation extraction across different backends.
    """

    def __init__(
        self,
        model_id: str,
        quantization: str = "none",
        device_map: str = "auto",
        dtype: str = "float16",
    ):
        """
        Initialize the backend.

        Args:
            model_id: HuggingFace model ID
            quantization: "none", "4bit", or "8bit"
            device_map: Device mapping strategy
            dtype: Model dtype ("float16", "bfloat16", "float32")
        """
        self.model_id = model_id
        self.quantization = quantization
        self.device_map = device_map
        self.dtype = dtype

        self._model = None
        self._tokenizer = None

    @property
    def model(self) -> Any:
        """Get the underlying model (lazy loaded)."""
        if self._model is None:
            self._load()
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer (lazy loaded)."""
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    @abstractmethod
    def _load(self) -> None:
        """Load the model and tokenizer. Called lazily on first use."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[list[str]] = None,
    ) -> GenerationOutput:
        """
        Generate text from a prompt.

        Args:
            prompt: The formatted prompt string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Optional stop sequences

        Returns:
            GenerationOutput with generated text and metadata
        """
        pass

    @abstractmethod
    def get_hidden_states(
        self,
        text: str,
        layers: Optional[list[int]] = None,
    ) -> dict[int, Tensor]:
        """
        Extract hidden states from a forward pass.

        Args:
            text: Input text
            layers: Layer indices to extract (None = all layers)

        Returns:
            Dictionary mapping layer index to hidden state tensor
            Shape: (seq_len, hidden_dim)
        """
        pass

    @abstractmethod
    def format_chat(
        self,
        system_prompt: str,
        user_turns: list[str],
        assistant_prefix: str = "",
    ) -> str:
        """
        Format a chat conversation into a prompt string.

        Args:
            system_prompt: System prompt
            user_turns: List of user messages
            assistant_prefix: Optional prefix for assistant response

        Returns:
            Formatted prompt string
        """
        pass

    def get_device(self) -> torch.device:
        """Get the device the model is on."""
        if hasattr(self.model, "device"):
            return self.model.device
        # For multi-GPU models, get device of first parameter
        return next(self.model.parameters()).device

    def count_parameters(self) -> int:
        """Count total model parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
