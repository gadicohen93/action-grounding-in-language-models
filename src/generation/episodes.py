"""
Episode generation using backend abstraction.

This module handles the actual generation of episodes, including:
- Model interaction via backends
- Episode labeling
- Batch generation with progress tracking
"""

import logging
import time
from datetime import datetime
from typing import Optional

from ..backends import get_backend
from ..config import get_config, get_secrets
from ..data import Episode, EpisodeCategory, ToolType
from .prompts import (
    SystemVariant,
    SocialPressure,
    Scenario,
    build_episode_config,
)

logger = logging.getLogger(__name__)


def log_section(title: str, char: str = "="):
    """Log a section header."""
    logger.info(char * 60)
    logger.info(f"  {title}")
    logger.info(char * 60)


def log_gpu_memory():
    """Log GPU memory usage if available."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"  GPU Memory: {allocated:.1f}GB allocated / {reserved:.1f}GB reserved / {total:.1f}GB total")
    except Exception:
        pass


class EpisodeGenerator:
    """
    Generator for creating labeled episodes.

    Uses a model backend to generate responses and a labeling function
    to categorize them.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        backend_type: Optional[str] = None,
        quantization: Optional[str] = None,
        labeling_method: str = "openai",
    ):
        """
        Initialize the episode generator.

        Args:
            model_id: Model to use (from config if None)
            backend_type: Backend type (from config if None)
            quantization: Quantization level (from config if None)
            labeling_method: Method for claim detection ("openai", "regex")
        """
        config = get_config()

        self.model_id = model_id or config.model.id
        self.backend_type = backend_type or config.model.backend
        self.quantization = quantization or config.model.quantization
        self.labeling_method = labeling_method

        # Get generation config
        self.temperature = config.generation.temperature
        self.max_tokens = config.generation.max_tokens
        self.top_p = config.generation.top_p
        self.seed = config.generation.seed

        # Initialize backend (lazy loaded)
        backend_class = get_backend(self.backend_type)
        self.backend = backend_class(
            model_id=self.model_id,
            quantization=self.quantization,
            device_map=config.model.device_map,
            dtype=config.model.dtype,
        )

        logger.info(f"Initialized EpisodeGenerator:")
        logger.info(f"  Model: {self.model_id}")
        logger.info(f"  Backend: {self.backend_type}")
        logger.info(f"  Quantization: {self.quantization}")
        logger.info(f"  Labeling: {self.labeling_method}")

    def generate(
        self,
        scenario: Scenario,
        variant: SystemVariant,
        pressure: SocialPressure,
        temperature: Optional[float] = None,
    ) -> Episode:
        """
        Generate a single episode.

        Args:
            scenario: Scenario to use
            variant: System prompt variant
            pressure: Social pressure condition
            temperature: Override temperature (uses config if None)

        Returns:
            Labeled Episode
        """
        # Build episode config
        config = build_episode_config(scenario, variant, pressure)

        # Generate response
        start_time = time.time()
        output = self.backend.generate(
            prompt=self.backend.format_chat(
                config["system_prompt"],
                config["user_turns"],
            ),
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        generation_time = time.time() - start_time

        # Detect tool usage and claims
        from ..labeling.tool_detection import detect_tool_call
        from ..labeling.claim_detection import detect_action_claim

        tool_call_result = detect_tool_call(
            output.text,
            tool_type=ToolType(config["tool_type"]),
        )

        claim_result = detect_action_claim(
            output.text,
            tool_type=ToolType(config["tool_type"]),
            method=self.labeling_method,
        )

        # Compute category
        category = Episode.compute_category(
            tool_used=tool_call_result["tool_used"],
            claims_action=claim_result["claims_action"],
        )

        # Map labeling method to Episode's expected format
        # Episode expects "llm" or "regex", but we use "openai" or "regex"
        claim_method = "llm" if self.labeling_method == "openai" else self.labeling_method

        # Build Episode object
        episode = Episode(
            tool_type=ToolType(config["tool_type"]),
            scenario=config["scenario"],
            system_variant=SystemVariant(config["system_variant"]),
            social_pressure=SocialPressure(config["social_pressure"]),
            system_prompt=config["system_prompt"],
            user_turns=config["user_turns"],
            assistant_reply=output.text,
            tool_used=tool_call_result["tool_used"],
            claims_action=claim_result["claims_action"],
            category=category,
            claim_detection_method=claim_method,
            claim_detection_confidence=claim_result.get("confidence"),
            claim_detection_reason=claim_result.get("reason"),
            model_id=self.model_id,
            generation_seed=self.seed,
            num_tokens_generated=output.num_tokens_generated,
            tool_call_raw=tool_call_result.get("raw_call"),
            tool_call_args=tool_call_result.get("args"),
        )

        return episode

    def generate_batch(
        self,
        conditions: list[dict],
        n_per_condition: int = 1,
        save_path: Optional[str] = None,
        verbose: bool = True,
        use_batch_labeling: bool = True,
        checkpoint_every: int = 100,
        checkpoint_path: Optional[str] = None,
    ) -> list[Episode]:
        """
        Generate a batch of episodes.

        Args:
            conditions: List of condition dicts from get_all_conditions()
            n_per_condition: Number of episodes per condition
            save_path: Optional path to save episodes (Parquet)
            verbose: Print progress
            use_batch_labeling: If True, batch OpenAI claim detection calls (much faster)
            checkpoint_every: Save checkpoint every N episodes (0 to disable)
            checkpoint_path: Path for checkpoint file (auto-generated if None)

        Returns:
            List of Episode objects
        """
        import json
        import os
        from tqdm import tqdm
        from collections import defaultdict

        total = len(conditions) * n_per_condition
        episodes = []

        pipeline_start = time.time()

        # Setup checkpointing
        if checkpoint_path is None and checkpoint_every > 0:
            checkpoint_path = str(save_path).replace('.parquet', '_checkpoint.jsonl') if save_path else './episode_checkpoint.jsonl'

        def save_checkpoint(data_list, path, msg=""):
            """Save intermediate results to JSONL."""
            try:
                with open(path, 'w') as f:
                    for item in data_list:
                        # Convert to serializable format
                        serializable = {
                            "output_text": item.get("output_text", ""),
                            "tool_type": item.get("tool_type").value if item.get("tool_type") else None,
                            "config": {k: v for k, v in item.get("config", {}).items() if k != "system_prompt"},
                            "num_tokens": item.get("output").num_tokens_generated if item.get("output") else 0,
                        }
                        f.write(json.dumps(serializable) + '\n')
                if msg:
                    logger.info(f"  💾 Checkpoint saved: {path} ({len(data_list)} items) {msg}")
            except Exception as e:
                logger.warning(f"  Failed to save checkpoint: {e}")

        if verbose:
            log_section("EPISODE GENERATION PIPELINE")
            logger.info(f"  Started at: {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"  Total episodes to generate: {total}")
            logger.info(f"  Conditions: {len(conditions)}")
            logger.info(f"  Episodes per condition: {n_per_condition}")
            logger.info(f"  Backend: {self.backend_type}")
            logger.info(f"  Model: {self.model_id}")
            logger.info(f"  Max tokens: {self.max_tokens}")
            logger.info(f"  Temperature: {self.temperature}")
            log_gpu_memory()
            if use_batch_labeling and self.labeling_method == "openai":
                logger.info("  Claim detection: Batched OpenAI (fast)")
            else:
                logger.info(f"  Claim detection: {self.labeling_method}")

        # Phase 1: Collect all prompts
        phase1_start = time.time()
        if verbose:
            log_section("PHASE 1: Preparing Prompts", "-")

        episode_data = []  # Store (episode_data, tool_type) for batch processing
        prompts_to_generate = []
        prompt_metadata = []  # Store metadata for each prompt

        for condition in conditions:
            # Build scenario object
            from .prompts import get_scenarios_for_tool, ToolType

            tool_type = ToolType(condition["tool_type"])
            scenarios = get_scenarios_for_tool(tool_type)
            scenario = next(
                (s for s in scenarios if s.name == condition["scenario"]),
                None
            )

            if scenario is None:
                logger.warning(f"Scenario not found: {condition['scenario']}")
                continue

            variant = SystemVariant(condition["system_variant"])
            pressure = SocialPressure(condition["social_pressure"])

            for _ in range(n_per_condition):
                try:
                    # Build episode config
                    config = build_episode_config(scenario, variant, pressure)

                    # Format prompt
                    prompt = self.backend.format_chat(
                        config["system_prompt"],
                        config["user_turns"],
                    )

                    prompts_to_generate.append(prompt)
                    prompt_metadata.append({
                        "config": config,
                        "tool_type": tool_type,
                    })
                except Exception as e:
                    logger.error(f"Failed to prepare episode: {e}")

        phase1_time = time.time() - phase1_start
        if verbose:
            logger.info(f"  Prepared {len(prompts_to_generate)} prompts in {phase1_time:.1f}s")
            # Show prompt length stats
            prompt_lens = [len(p) for p in prompts_to_generate]
            logger.info(f"  Prompt lengths: min={min(prompt_lens)}, max={max(prompt_lens)}, avg={sum(prompt_lens)/len(prompt_lens):.0f} chars")

        # Phase 2: Batch generate all prompts
        phase2_start = time.time()
        if verbose:
            log_section("PHASE 2: Model Generation", "-")
            logger.info(f"  Generating {len(prompts_to_generate)} responses...")

        # Check if backend supports batch generation (works for BOTH vLLM and PyTorch!)
        if hasattr(self.backend, 'generate_batch'):
            # Use batch generation (much faster!)
            if self.backend_type == "vllm":
                logger.info("  Using vLLM continuous batching (all prompts at once)")
            else:
                logger.info("  Using PyTorch batch generation (batch_size=32)")

            outputs = self.backend.generate_batch(
                prompts_to_generate,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                batch_size=32,  # Ignored by vLLM (uses continuous batching)
                verbose=verbose,
            )
        else:
            # Fallback to sequential (slow!)
            logger.warning("  No batch generation available - using sequential (SLOW)")
            outputs = []
            progress = tqdm(prompts_to_generate, desc="Generating") if verbose else prompts_to_generate
            for prompt in progress:
                output = self.backend.generate(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
                outputs.append(output)

        phase2_time = time.time() - phase2_start
        if verbose:
            logger.info(f"  Generation complete in {phase2_time:.1f}s")
            logger.info(f"  Speed: {len(outputs)/phase2_time:.1f} responses/sec")
            log_gpu_memory()

        # Process outputs (tool detection)
        if verbose:
            log_section("PHASE 3: Tool Detection", "-")
            logger.info(f"  Detecting tool calls in {len(outputs)} responses...")

        phase3_start = time.time()
        from ..labeling.tool_detection import detect_tool_call

        for i, (output, metadata) in enumerate(zip(outputs, prompt_metadata)):
            try:
                config = metadata["config"]
                tool_type = metadata["tool_type"]

                # Detect tool usage (fast, regex-based)
                tool_call_result = detect_tool_call(
                    output.text,
                    tool_type=tool_type,
                )

                # Store for batch claim detection
                episode_data.append({
                    "output_text": output.text,
                    "tool_type": tool_type,
                    "tool_call_result": tool_call_result,
                    "config": config,
                    "output": output,
                })

            except Exception as e:
                logger.error(f"Failed to process episode {i}: {e}")

        phase3_time = time.time() - phase3_start
        if verbose:
            tool_used_count = sum(1 for d in episode_data if d["tool_call_result"]["tool_used"])
            logger.info(f"  Tool detection complete in {phase3_time:.1f}s")
            logger.info(f"  Tools used: {tool_used_count}/{len(episode_data)} ({100*tool_used_count/len(episode_data):.1f}%)")

        # Checkpoint after generation (most expensive phase)
        if checkpoint_every > 0 and checkpoint_path and len(episode_data) > 0:
            save_checkpoint(episode_data, checkpoint_path, f"after generation")

        # Phase 4: Batch claim detection
        phase4_start = time.time()
        if verbose:
            log_section("PHASE 4: Claim Detection", "-")

        if use_batch_labeling and self.labeling_method == "openai":
            if verbose:
                logger.info("  Using batched OpenAI API calls...")

            from ..labeling.claim_detection import detect_action_claims_batch

            # Group by tool_type for batch processing
            by_tool_type = defaultdict(list)
            indices_by_tool_type = defaultdict(list)

            for idx, data in enumerate(episode_data):
                tool_type = data["tool_type"]
                by_tool_type[tool_type].append(data["output_text"])
                indices_by_tool_type[tool_type].append(idx)

            if verbose:
                for tt, texts in by_tool_type.items():
                    logger.info(f"    {tt.value}: {len(texts)} texts")

            # Batch detect claims for each tool type
            claim_results = {}
            for tool_type, texts in by_tool_type.items():
                if verbose:
                    logger.info(f"  Detecting claims for {tool_type.value}...")
                batch_start = time.time()
                results = detect_action_claims_batch(
                    texts=texts,
                    tool_type=tool_type,
                    method="openai",
                )
                batch_time = time.time() - batch_start
                if verbose:
                    claims_count = sum(1 for r in results if r.get("claims_action"))
                    logger.info(f"    {len(results)} processed in {batch_time:.1f}s ({claims_count} claims)")

                # Map results back to original indices
                for idx, result in zip(indices_by_tool_type[tool_type], results):
                    claim_results[idx] = result

        else:
            # Sequential claim detection (fallback)
            if verbose:
                logger.info(f"  Using sequential {self.labeling_method} detection...")
            claim_results = {}
            from ..labeling.claim_detection import detect_action_claim
            progress = tqdm(episode_data, desc="Detecting claims") if verbose else episode_data

            for idx, data in enumerate(progress if not verbose else episode_data):
                claim_results[idx] = detect_action_claim(
                    data["output_text"],
                    tool_type=data["tool_type"],
                    method=self.labeling_method,
                )

        phase4_time = time.time() - phase4_start
        if verbose:
            claims_count = sum(1 for r in claim_results.values() if r.get("claims_action"))
            logger.info(f"  Claim detection complete in {phase4_time:.1f}s")
            logger.info(f"  Claims detected: {claims_count}/{len(claim_results)} ({100*claims_count/len(claim_results):.1f}%)")

        # Phase 5: Build Episode objects
        phase5_start = time.time()
        if verbose:
            log_section("PHASE 5: Building Episodes", "-")
            logger.info(f"  Creating {len(episode_data)} Episode objects...")

        for idx, data in enumerate(episode_data):
            claim_result = claim_results[idx]
            tool_call_result = data["tool_call_result"]
            config = data["config"]
            output = data["output"]

            # Compute category
            category = Episode.compute_category(
                tool_used=tool_call_result["tool_used"],
                claims_action=claim_result["claims_action"],
            )

            # Map labeling method to Episode's expected format
            claim_method = "llm" if self.labeling_method == "openai" else self.labeling_method

            episode = Episode(
                tool_type=ToolType(config["tool_type"]),
                scenario=config["scenario"],
                system_variant=SystemVariant(config["system_variant"]),
                social_pressure=SocialPressure(config["social_pressure"]),
                system_prompt=config["system_prompt"],
                user_turns=config["user_turns"],
                assistant_reply=output.text,
                tool_used=tool_call_result["tool_used"],
                claims_action=claim_result["claims_action"],
                category=category,
                claim_detection_method=claim_method,
                claim_detection_confidence=claim_result.get("confidence"),
                claim_detection_reason=claim_result.get("reason"),
                model_id=self.model_id,
                generation_seed=self.seed,
                num_tokens_generated=output.num_tokens_generated,
                tool_call_raw=tool_call_result.get("raw_call"),
                tool_call_args=tool_call_result.get("args"),
            )

            episodes.append(episode)

        phase5_time = time.time() - phase5_start
        total_time = time.time() - pipeline_start

        if verbose:
            logger.info(f"  Episode creation complete in {phase5_time:.1f}s")

            # Final summary
            log_section("PIPELINE COMPLETE")
            total_generated = len(episodes)
            fake_count = sum(1 for e in episodes if e.is_fake())
            fake_rate = fake_count / total_generated if total_generated > 0 else 0

            # Category breakdown
            from collections import Counter
            categories = Counter(
                e.category.value if hasattr(e.category, 'value') else e.category
                for e in episodes
            )

            logger.info(f"  Total episodes: {total_generated}")
            logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            logger.info(f"  Speed: {total_generated/total_time:.1f} episodes/sec")
            logger.info(f"")
            logger.info(f"  CATEGORY BREAKDOWN:")
            for cat, count in sorted(categories.items()):
                logger.info(f"    {cat}: {count} ({100*count/total_generated:.1f}%)")
            logger.info(f"")
            logger.info(f"  FAKE RATE: {fake_rate:.1%} ({fake_count}/{total_generated})")
            logger.info(f"")
            logger.info(f"  TIME BREAKDOWN:")
            logger.info(f"    Phase 1 (Prompts):    {phase1_time:6.1f}s ({100*phase1_time/total_time:5.1f}%)")
            logger.info(f"    Phase 2 (Generation): {phase2_time:6.1f}s ({100*phase2_time/total_time:5.1f}%)")
            logger.info(f"    Phase 3 (Tool Det.):  {phase3_time:6.1f}s ({100*phase3_time/total_time:5.1f}%)")
            logger.info(f"    Phase 4 (Claims):     {phase4_time:6.1f}s ({100*phase4_time/total_time:5.1f}%)")
            logger.info(f"    Phase 5 (Episodes):   {phase5_time:6.1f}s ({100*phase5_time/total_time:5.1f}%)")
            log_gpu_memory()

        # Save if requested
        if save_path:
            from ..data.io import save_episodes

            save_episodes(episodes, save_path)
            logger.info(f"Saved episodes to: {save_path}")

        return episodes

    def unload(self):
        """Unload model from memory."""
        self.backend.unload()


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_episode(
    scenario: Scenario,
    variant: SystemVariant,
    pressure: SocialPressure,
    model_id: Optional[str] = None,
    temperature: Optional[float] = None,
    labeling_method: str = "openai",
) -> Episode:
    """
    Generate a single episode (convenience function).

    Args:
        scenario: Scenario to use
        variant: System prompt variant
        pressure: Social pressure condition
        model_id: Model to use (from config if None)
        temperature: Sampling temperature (from config if None)
        labeling_method: Claim detection method

    Returns:
        Labeled Episode
    """
    generator = EpisodeGenerator(
        model_id=model_id,
        labeling_method=labeling_method,
    )

    try:
        episode = generator.generate(scenario, variant, pressure, temperature)
    finally:
        generator.unload()

    return episode


def generate_batch(
    conditions: list[dict],
    n_per_condition: int = 1,
    model_id: Optional[str] = None,
    labeling_method: str = "openai",
    save_path: Optional[str] = None,
    verbose: bool = True,
    use_batch_labeling: bool = True,
) -> list[Episode]:
    """
    Generate a batch of episodes (convenience function).

    Args:
        conditions: List of condition dicts
        n_per_condition: Episodes per condition
        model_id: Model to use (from config if None)
        labeling_method: Claim detection method
        save_path: Optional path to save
        verbose: Print progress
        use_batch_labeling: If True, batch OpenAI claim detection calls (much faster)

    Returns:
        List of Episodes
    """
    generator = EpisodeGenerator(
        model_id=model_id,
        labeling_method=labeling_method,
    )

    try:
        episodes = generator.generate_batch(
            conditions,
            n_per_condition,
            save_path,
            verbose,
            use_batch_labeling=use_batch_labeling,
        )
    finally:
        generator.unload()

    return episodes
