"""
Model backends for the action-grounding research project.

This module provides a unified interface for interacting with language models,
abstracting away the differences between PyTorch/Transformers and vLLM.
"""

from .base import ModelBackend, GenerationOutput
from .pytorch import PyTorchBackend

__all__ = ["ModelBackend", "GenerationOutput", "PyTorchBackend"]

# Try to import vLLM backend (optional dependency)
try:
    from .vllm import VLLMBackend
    __all__.append("VLLMBackend")
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    VLLMBackend = None


def get_backend(backend_type: str = "pytorch") -> type[ModelBackend]:
    """
    Get the backend class by name.

    Args:
        backend_type: "pytorch" or "vllm"

    Returns:
        Backend class (not instance)

    Raises:
        ValueError: If backend type is unknown or vLLM requested but not installed
    """
    backends = {
        "pytorch": PyTorchBackend,
    }

    # Add vLLM if available
    if VLLM_AVAILABLE and VLLMBackend is not None:
        backends["vllm"] = VLLMBackend

    if backend_type not in backends:
        available = list(backends.keys())
        if backend_type == "vllm" and not VLLM_AVAILABLE:
            raise ValueError(
                f"vLLM backend requested but not installed. "
                f"Install with: pip install vllm\n"
                f"Available backends: {available}"
            )
        raise ValueError(
            f"Unknown backend: {backend_type}. Available: {available}"
        )

    return backends[backend_type]
