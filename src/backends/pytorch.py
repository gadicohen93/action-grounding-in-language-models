"""
PyTorch/Transformers backend implementation.

Uses HuggingFace Transformers for model loading and inference.
Supports quantization via bitsandbytes.
"""

import logging
from typing import Optional

import torch
from torch import Tensor

from .base import ModelBackend, GenerationOutput

logger = logging.getLogger(__name__)


class PyTorchBackend(ModelBackend):
    """
    PyTorch/Transformers backend for language models.

    Supports:
    - Full precision (float16, bfloat16, float32)
    - 8-bit quantization via bitsandbytes
    - 4-bit quantization via bitsandbytes
    - Multi-GPU via device_map="auto"
    """

    def _load(self) -> None:
        """Load model and tokenizer from HuggingFace."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading model: {self.model_id}")
        logger.info(f"  Quantization: {self.quantization}")
        logger.info(f"  Device map: {self.device_map}")
        logger.info(f"  Dtype: {self.dtype}")

        # Configure dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float16)

        # Configure quantization
        load_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }

        if self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,  # Allow CPU offload if GPU is tight
            )
            load_kwargs["quantization_config"] = quantization_config
            logger.info("  Using 8-bit quantization (with CPU offload enabled)")
        elif self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = quantization_config
            logger.info("  Using 4-bit quantization")
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **load_kwargs,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(f"Model loaded. Parameters: {self.count_parameters():,}")

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
            GenerationOutput with generated text
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=4096,
        )
        input_ids = inputs["input_ids"].to(self.get_device())
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.get_device())

        # Set up generation config
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        # Handle stop sequences
        if stop_sequences:
            # Convert stop sequences to token IDs
            stop_token_ids = []
            for seq in stop_sequences:
                ids = self.tokenizer.encode(seq, add_special_tokens=False)
                if ids:
                    stop_token_ids.append(ids[0])
            if stop_token_ids:
                gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + stop_token_ids

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # Extract generated portion
        generated_ids = output_ids[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Strip stop sequences from output
        if stop_sequences:
            for seq in stop_sequences:
                if seq in generated_text:
                    generated_text = generated_text.split(seq)[0]

        return GenerationOutput(
            text=generated_text,
            input_ids=input_ids[0],
            output_ids=generated_ids,
            num_tokens_generated=len(generated_ids),
        )

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[list[str]] = None,
        batch_size: int = 8,
        verbose: bool = True,
    ) -> list[GenerationOutput]:
        """
        Generate text from multiple prompts in batches (much faster than sequential).

        Args:
            prompts: List of prompt strings
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Optional stop sequences
            batch_size: Number of prompts to process at once
            verbose: Show progress bar

        Returns:
            List of GenerationOutput objects
        """
        from tqdm import tqdm
        
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        results = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        logger.info(f"Starting batch generation: {len(prompts)} prompts in {total_batches} batches (batch_size={batch_size})")
        start_time = time.time()
        
        # Process in batches with progress bar
        iterator = range(0, len(prompts), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Generating batches", total=total_batches, unit="batch")
        
        for batch_idx, i in enumerate(iterator, 1):
            batch_prompts = prompts[i:i + batch_size]
            batch_start = time.time()
            
            # Log progress more frequently
            if verbose:
                if batch_idx <= 5 or batch_idx % 5 == 0:
                    logger.info(f"Processing batch {batch_idx}/{total_batches} ({batch_idx*100//total_batches}%)")
            
            # Tokenize batch with padding
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )
            input_ids = inputs["input_ids"].to(self.get_device())
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.get_device())

            # Set up generation config
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            # Handle stop sequences
            if stop_sequences:
                stop_token_ids = []
                for seq in stop_sequences:
                    ids = self.tokenizer.encode(seq, add_special_tokens=False)
                    if ids:
                        stop_token_ids.append(ids[0])
                if stop_token_ids:
                    gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + stop_token_ids

            # Generate batch
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            
            batch_time = time.time() - batch_start
            # Log completion more frequently
            if verbose:
                if batch_idx <= 5 or batch_idx % 5 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / batch_idx
                    remaining = avg_time * (total_batches - batch_idx)
                    logger.info(f"Batch {batch_idx}/{total_batches} completed in {batch_time:.2f}s | "
                              f"Elapsed: {elapsed/60:.1f}min | "
                              f"ETA: {remaining/60:.1f}min | "
                              f"Speed: {batch_size/batch_time:.1f} prompts/s")

            # Extract generated portions for each prompt
            for j, prompt in enumerate(batch_prompts):
                # Find where prompt ends (accounting for left-padding)
                # Count padding tokens from the left
                pad_count = (input_ids[j] == self.tokenizer.pad_token_id).long().argmin().item()
                # Get the actual prompt length (non-padded tokens)
                prompt_len = attention_mask[j].sum().item() if attention_mask is not None else input_ids[j].ne(self.tokenizer.pad_token_id).sum().item()
                # Input ends at: pad_count + prompt_len
                input_end = pad_count + prompt_len

                # Extract generated portion (everything after the prompt)
                generated_ids = output_ids[j, input_end:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Strip stop sequences
                if stop_sequences:
                    for seq in stop_sequences:
                        if seq in generated_text:
                            generated_text = generated_text.split(seq)[0]

                results.append(GenerationOutput(
                    text=generated_text,
                    input_ids=input_ids[j],
                    output_ids=generated_ids,
                    num_tokens_generated=len(generated_ids),
                ))
        
        total_time = time.time() - start_time
        logger.info(f"Batch generation complete: {len(results)} outputs in {total_time/60:.1f} minutes "
                   f"({total_time/len(prompts):.2f}s per prompt)")

        return results

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
            Shape per tensor: (seq_len, hidden_dim)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=4096,
        )
        input_ids = inputs["input_ids"].to(self.get_device())
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.get_device())

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract hidden states
        # hidden_states is a tuple of (n_layers + 1) tensors
        # Index 0 is embeddings, indices 1-n are layer outputs
        hidden_states = outputs.hidden_states

        # Determine which layers to return
        n_layers = len(hidden_states) - 1  # Exclude embedding layer
        if layers is None:
            layers = list(range(n_layers))
        else:
            # Handle negative indices
            layers = [l if l >= 0 else n_layers + l for l in layers]

        # Build result dictionary
        result = {}
        for layer_idx in layers:
            if 0 <= layer_idx < n_layers:
                # hidden_states[layer_idx + 1] because index 0 is embeddings
                result[layer_idx] = hidden_states[layer_idx + 1][0].cpu()

        return result

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
            # (the last turn is what we're generating)
            if i < len(user_turns) - 1:
                messages.append({"role": "assistant", "content": ""})

        # Try using the tokenizer's chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if assistant_prefix:
                    prompt += assistant_prefix
                return prompt
            except Exception as e:
                logger.warning(f"Chat template failed: {e}. Falling back to manual format.")

        # Fallback: Manual formatting for common models
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

        # Mistral v0.2 uses [INST] tags
        # System prompt goes in first user turn
        first_user = f"{system_prompt}\n\n{user_turns[0]}" if system_prompt else user_turns[0]
        parts.append(f"<s>[INST] {first_user} [/INST]")

        # Add remaining turns
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

    def get_num_layers(self) -> int:
        """Get the number of transformer layers."""
        if hasattr(self.model.config, "num_hidden_layers"):
            return self.model.config.num_hidden_layers
        elif hasattr(self.model.config, "n_layer"):
            return self.model.config.n_layer
        else:
            # Fallback: count hidden states
            dummy_states = self.get_hidden_states("test")
            return len(dummy_states)

    def get_hidden_size(self) -> int:
        """Get the hidden dimension size."""
        if hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        elif hasattr(self.model.config, "n_embd"):
            return self.model.config.n_embd
        else:
            # Fallback: check first layer output
            dummy_states = self.get_hidden_states("test")
            first_layer = list(dummy_states.values())[0]
            return first_layer.shape[-1]
