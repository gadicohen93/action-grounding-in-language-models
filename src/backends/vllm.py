"""
vLLM backend implementation.

Uses vLLM for fast inference with PagedAttention and continuous batching.
Note: vLLM doesn't support activation extraction, so this backend is
optimized for generation only.
"""

import logging
from typing import Optional

import torch
from torch import Tensor

from .base import ModelBackend, GenerationOutput

logger = logging.getLogger(__name__)


class VLLMBackend(ModelBackend):
    """
    vLLM backend for fast language model inference.

    Supports:
    - Fast inference with PagedAttention (2-3x faster than PyTorch)
    - Continuous batching for high throughput
    - Multi-GPU support

    Limitations:
    - Activation extraction not supported (use PyTorch backend for that)
    - Quantization options limited (vLLM handles this internally)
    """

    def _load(self) -> None:
        """Load model using vLLM."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )

        logger.info(f"Loading model with vLLM: {self.model_id}")
        logger.info(f"  Device map: {self.device_map}")
        logger.info(f"  Dtype: {self.dtype}")

        # Configure dtype for vLLM
        dtype_map = {
            "float16": "half",
            "bfloat16": "bfloat16",
            "float32": "float",
        }
        vllm_dtype = dtype_map.get(self.dtype, "half")

        # vLLM configuration
        # Note: vLLM handles quantization internally, so we ignore the quantization param
        # but log a warning if it's set
        if self.quantization != "none":
            logger.warning(
                f"vLLM doesn't support explicit quantization '{self.quantization}'. "
                "Using vLLM's internal optimizations instead."
            )

        # Configure tensor parallelism for multi-GPU
        tensor_parallel_size = 1
        if self.device_map == "auto" and torch.cuda.device_count() > 1:
            tensor_parallel_size = torch.cuda.device_count()
            logger.info(f"  Using tensor parallelism: {tensor_parallel_size} GPUs")

        # Load model with vLLM
        self._model = LLM(
            model=self.model_id,
            dtype=vllm_dtype,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,  # Use 90% of GPU memory
        )

        # Load tokenizer separately (vLLM doesn't expose it directly)
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("Model loaded with vLLM (fast inference mode)")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[list[str]] = None,
    ) -> GenerationOutput:
        """
        Generate text from a prompt using vLLM.

        Args:
            prompt: The formatted prompt string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Optional stop sequences

        Returns:
            GenerationOutput with generated text
        """
        from vllm import SamplingParams

        # Configure sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            stop=stop_sequences if stop_sequences else None,
        )

        # Generate (vLLM expects a list of prompts)
        outputs = self._model.generate([prompt], sampling_params)

        # Extract result (vLLM returns a list of outputs)
        output = outputs[0]
        generated_text = output.outputs[0].text

        # Count tokens
        num_tokens = len(output.outputs[0].token_ids)

        # Tokenize for compatibility (for input_ids/output_ids)
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        output_ids = self._tokenizer.encode(generated_text, return_tensors="pt")

        return GenerationOutput(
            text=generated_text,
            input_ids=input_ids[0] if len(input_ids) > 0 else None,
            output_ids=output_ids[0] if len(output_ids) > 0 else None,
            num_tokens_generated=num_tokens,
        )

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[list[str]] = None,
        batch_size: int = 64,  # vLLM handles this internally
        verbose: bool = True,
    ) -> list[GenerationOutput]:
        """
        Generate text from multiple prompts using vLLM's continuous batching.

        vLLM is optimized for batch generation - it uses PagedAttention and
        continuous batching internally, so we pass ALL prompts at once for
        maximum throughput.

        Args:
            prompts: List of prompt strings
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Optional stop sequences
            batch_size: Ignored (vLLM handles batching internally)
            verbose: Show progress info

        Returns:
            List of GenerationOutput objects
        """
        import time
        from vllm import SamplingParams

        if verbose:
            logger.info(f"=" * 60)
            logger.info(f"vLLM BATCH GENERATION")
            logger.info(f"=" * 60)
            logger.info(f"  Total prompts: {len(prompts)}")
            logger.info(f"  Max tokens: {max_tokens}")
            logger.info(f"  Temperature: {temperature}")
            logger.info(f"  vLLM uses continuous batching (no explicit batch size)")

        # Configure sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            stop=stop_sequences if stop_sequences else None,
        )

        # CRITICAL: Pass ALL prompts to vLLM at once
        # vLLM's scheduler handles batching internally with continuous batching
        start_time = time.time()
        if verbose:
            logger.info(f"  Starting generation at {time.strftime('%H:%M:%S')}")
            logger.info(f"  This may take a minute... (vLLM optimizes internally)")

        outputs = self._model.generate(prompts, sampling_params)

        elapsed = time.time() - start_time
        if verbose:
            logger.info(f"  Generation complete in {elapsed:.1f}s")
            logger.info(f"  Throughput: {len(prompts)/elapsed:.1f} prompts/sec")
            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            logger.info(f"  Total tokens generated: {total_tokens:,}")
            logger.info(f"  Token throughput: {total_tokens/elapsed:.0f} tokens/sec")

        # Convert to GenerationOutput objects
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)

            # Tokenize for compatibility (batched for speed)
            input_ids = self._tokenizer.encode(prompts[i], return_tensors="pt")
            output_ids = self._tokenizer.encode(generated_text, return_tensors="pt")

            results.append(GenerationOutput(
                text=generated_text,
                input_ids=input_ids[0] if len(input_ids) > 0 else None,
                output_ids=output_ids[0] if len(output_ids) > 0 else None,
                num_tokens_generated=num_tokens,
            ))

        if verbose:
            logger.info(f"  Converted {len(results)} outputs")
            logger.info(f"=" * 60)

        return results

    def get_hidden_states(
        self,
        text: str,
        layers: Optional[list[int]] = None,
    ) -> dict[int, Tensor]:
        """
        Extract hidden states from a forward pass.

        NOTE: vLLM doesn't support activation extraction.
        This method raises an error suggesting to use PyTorch backend instead.

        Args:
            text: Input text
            layers: Layer indices to extract (ignored)

        Returns:
            Dictionary mapping layer index to hidden state tensor

        Raises:
            NotImplementedError: vLLM doesn't support activation extraction
        """
        raise NotImplementedError(
            "vLLM backend doesn't support activation extraction. "
            "Use PyTorch backend (backend='pytorch') for activation extraction."
        )

    def format_chat(
        self,
        system_prompt: str,
        user_turns: list[str],
        assistant_prefix: str = "",
    ) -> str:
        """
        Format a chat conversation into a prompt string.

        Uses the tokenizer's built-in chat template when available.

        Args:
            system_prompt: System prompt
            user_turns: List of user messages
            assistant_prefix: Optional prefix for assistant response

        Returns:
            Formatted prompt string
        """
        # Build messages list
        messages = []

        # Add system message
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add user/assistant turns
        for i, user_msg in enumerate(user_turns):
            messages.append({"role": "user", "content": user_msg})
            # Add empty assistant message for all but the last turn
            if i < len(user_turns) - 1:
                messages.append({"role": "assistant", "content": ""})

        # Try using the tokenizer's chat template
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if assistant_prefix:
                    prompt += assistant_prefix
                return prompt
            except Exception as e:
                logger.warning(f"Chat template failed: {e}. Falling back to manual format.")

        # Fallback: Manual formatting
        return self._format_chat_manual(system_prompt, user_turns, assistant_prefix)

    def _format_chat_manual(
        self,
        system_prompt: str,
        user_turns: list[str],
        assistant_prefix: str = "",
    ) -> str:
        """
        Manual chat formatting fallback.

        Supports Mistral, Llama 3, and generic instruction format.
        """
        model_lower = self.model_id.lower()

        if "mistral" in model_lower:
            return self._format_mistral(system_prompt, user_turns, assistant_prefix)
        elif "llama-3" in model_lower or "llama3" in model_lower:
            return self._format_llama3(system_prompt, user_turns, assistant_prefix)
        else:
            return self._format_generic(system_prompt, user_turns, assistant_prefix)

    def _format_mistral(
        self,
        system_prompt: str,
        user_turns: list[str],
        assistant_prefix: str = "",
    ) -> str:
        """Format for Mistral Instruct."""
        parts = []

        first_user = f"{system_prompt}\n\n{user_turns[0]}" if system_prompt else user_turns[0]
        parts.append(f"<s>[INST] {first_user} [/INST]")

        for i, user_msg in enumerate(user_turns[1:], start=1):
            parts.append(f"</s>[INST] {user_msg} [/INST]")

        prompt = "".join(parts)
        if assistant_prefix:
            prompt += assistant_prefix
        return prompt

    def _format_llama3(
        self,
        system_prompt: str,
        user_turns: list[str],
        assistant_prefix: str = "",
    ) -> str:
        """Format for Llama 3 Instruct."""
        parts = ["<|begin_of_text|>"]

        if system_prompt:
            parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>")

        for user_msg in user_turns:
            parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>")

        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        prompt = "".join(parts)
        if assistant_prefix:
            prompt += assistant_prefix
        return prompt

    def _format_generic(
        self,
        system_prompt: str,
        user_turns: list[str],
        assistant_prefix: str = "",
    ) -> str:
        """Generic instruction format."""
        parts = []

        if system_prompt:
            parts.append(f"### System:\n{system_prompt}\n")

        for user_msg in user_turns:
            parts.append(f"### User:\n{user_msg}\n")

        parts.append("### Assistant:\n")

        prompt = "\n".join(parts)
        if assistant_prefix:
            prompt += assistant_prefix
        return prompt

    def get_device(self) -> torch.device:
        """Get the device the model is on."""
        # vLLM manages devices internally, return CUDA device if available
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def count_parameters(self) -> int:
        """Count total model parameters."""
        # vLLM doesn't expose model directly, return approximate count
        # This is a rough estimate based on model config
        if hasattr(self._model, "llm_engine"):
            # Try to get from engine if available
            return 0  # vLLM doesn't expose this easily
        return 0

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




