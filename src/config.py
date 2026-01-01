"""
Configuration management for the action-grounding research project.

Uses Pydantic for validation and YAML for configuration files.
Supports environment variable overrides via .env files.
"""

from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class ModelConfig(BaseModel):
    """Configuration for the language model."""

    id: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        description="HuggingFace model ID"
    )
    backend: Literal["pytorch", "vllm"] = Field(
        default="pytorch",
        description="Model backend to use"
    )
    quantization: Literal["none", "4bit", "8bit"] = Field(
        default="8bit",
        description="Quantization level for memory efficiency"
    )
    device_map: str = Field(
        default="auto",
        description="Device mapping strategy"
    )
    dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="float16",
        description="Model dtype"
    )


class GenerationConfig(BaseModel):
    """Configuration for episode generation."""

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum tokens to generate"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class DataConfig(BaseModel):
    """Configuration for data paths."""

    raw_dir: Path = Field(
        default=Path("./data/raw"),
        description="Directory for raw episode data"
    )
    processed_dir: Path = Field(
        default=Path("./data/processed"),
        description="Directory for processed data (activations, etc.)"
    )
    figures_dir: Path = Field(
        default=Path("./figures"),
        description="Directory for output figures"
    )

    @field_validator("raw_dir", "processed_dir", "figures_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class LabelingConfig(BaseModel):
    """Configuration for claim labeling."""

    method: Literal["openai", "regex", "local"] = Field(
        default="openai",
        description="Method for detecting action claims"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for LLM-based labeling"
    )
    batch_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Batch size for LLM labeling"
    )


class ExperimentConfig(BaseModel):
    """Configuration for experiment parameters."""

    n_episodes_per_condition: int = Field(
        default=10,
        ge=1,
        description="Number of episodes per experimental condition (full mode)"
    )
    n_episodes_per_condition_fast: int = Field(
        default=2,
        ge=1,
        description="Number of episodes per condition in fast mode"
    )
    tools: list[str] = Field(
        default=["escalate", "search", "sendMessage"],
        description="Tool types to test"
    )
    system_variants: list[str] = Field(
        default=["A_STRICT", "B_DILUTED", "C_CONFLICTING"],
        description="System prompt variants"
    )
    social_pressures: list[str] = Field(
        default=["NEUTRAL", "STRESSED", "DEMAND", "VALIDATION", "APPEASE"],
        description="Social pressure conditions"
    )


class ExtractionConfig(BaseModel):
    """Configuration for activation extraction."""

    positions: list[str] = Field(
        default=["first_assistant", "mid_response", "before_tool"],
        description="Token positions to extract activations from"
    )
    layers: list[int] = Field(
        default=[0, 8, 16, 24, 31],
        description="Layer indices to extract (0-indexed)"
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        description="Batch size for activation extraction"
    )


class ProbeConfig(BaseModel):
    """Configuration for probe training."""

    regularization: float = Field(
        default=1.0,
        gt=0,
        description="L2 regularization strength (C parameter)"
    )
    n_folds: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of cross-validation folds"
    )
    bootstrap_samples: int = Field(
        default=1000,
        ge=100,
        description="Number of bootstrap samples for confidence intervals"
    )


class SteeringConfig(BaseModel):
    """Configuration for steering vector experiments."""

    alphas: list[float] = Field(
        default=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        description="Steering strengths to test"
    )
    n_samples_per_alpha: int = Field(
        default=50,
        ge=10,
        description="Number of samples per steering strength"
    )
    target_layer: int = Field(
        default=16,
        ge=0,
        description="Layer to apply steering at"
    )


class Config(BaseModel):
    """Root configuration object."""

    fast_mode: bool = Field(
        default=True,
        description="Enable fast mode (~10 min) vs full mode (~60 min)"
    )
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    steering: SteeringConfig = Field(default_factory=SteeringConfig)

    def get_n_episodes_per_condition(self) -> int:
        """Get effective episodes per condition based on fast_mode."""
        if self.fast_mode:
            return self.experiment.n_episodes_per_condition_fast
        return self.experiment.n_episodes_per_condition

    def get_total_episodes(self) -> int:
        """Calculate total episodes based on conditions and mode."""
        n_conditions = (
            len(self.experiment.tools) *
            len(self.experiment.system_variants) *
            len(self.experiment.social_pressures)
        )
        return n_conditions * self.get_n_episodes_per_condition()

    def log_mode_info(self) -> str:
        """Return a string describing the current mode."""
        mode = "FAST" if self.fast_mode else "FULL"
        n_per = self.get_n_episodes_per_condition()
        total = self.get_total_episodes()
        return f"{mode} MODE: {n_per} eps/condition = {total} total episodes"

    @property
    def figures_dir(self) -> Path:
        """Convenience property for figures directory."""
        return self.data.figures_dir

    @property
    def raw_dir(self) -> Path:
        """Convenience property for raw data directory."""
        return self.data.raw_dir

    @property
    def processed_dir(self) -> Path:
        """Convenience property for processed data directory."""
        return self.data.processed_dir

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data) if data else cls()

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


class Secrets(BaseSettings):
    """
    Environment-based secrets configuration.

    Load from .env file or environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    hf_token: Optional[str] = Field(
        default=None,
        alias="HF_TOKEN",
        description="HuggingFace API token"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        alias="OPENAI_API_KEY",
        description="OpenAI API key"
    )


# Global configuration instance (lazy loaded)
_config: Optional[Config] = None
_secrets: Optional[Secrets] = None


def get_config(config_path: Optional[str | Path] = None) -> Config:
    """
    Get the global configuration instance.

    Args:
        config_path: Optional path to config.yaml. If not provided,
                    looks for config.yaml in the project root.

    Returns:
        Config instance
    """
    global _config

    if _config is None:
        if config_path is None:
            # Look for config.yaml in common locations
            candidates = [
                Path("config.yaml"),
                Path("../config.yaml"),
                Path(__file__).parent.parent / "config.yaml",
            ]
            for candidate in candidates:
                if candidate.exists():
                    config_path = candidate
                    break

        if config_path and Path(config_path).exists():
            _config = Config.from_yaml(config_path)
        else:
            # Use defaults
            _config = Config()

    return _config


def get_secrets() -> Secrets:
    """Get the global secrets instance."""
    global _secrets

    if _secrets is None:
        _secrets = Secrets()

    return _secrets


def reset_config() -> None:
    """Reset global configuration (useful for testing)."""
    global _config, _secrets
    _config = None
    _secrets = None
