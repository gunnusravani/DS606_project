"""
HH-RLHF Dataset Loading and Formatting

This module handles:
1. Loading HH-RLHF dataset from HuggingFace Hub
2. Formatting for SFT (Supervised Fine-Tuning)
3. Formatting for DPO (Direct Preference Optimization)
4. Tokenization and padding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: LOAD DATASET FROM HUGGINGFACE HUB
# ============================================================================

def load_hh_rlhf_dataset(
    split: str = "train",
    num_samples: Optional[int] = None,
) -> Dataset:
    """
    Load HH-RLHF dataset from HuggingFace Hub.
    
    Args:
        split: Which split to load ("train" or "test")
        num_samples: Limit number of samples (useful for testing)
    
    Returns:
        Dataset with 'prompt', 'chosen', 'rejected' columns
        
    Example:
        >>> dataset = load_hh_rlhf_dataset(split="train", num_samples=100)
        >>> print(dataset[0].keys())
        dict_keys(['prompt', 'chosen', 'rejected', ...])
    """
    logger.info(f"Loading HH-RLHF dataset ({split} split)")
    
    # Download dataset from HuggingFace Hub
    # Note: First download takes ~30 seconds, then cached locally
    dataset = load_dataset("Anthropic/hh-rlhf", split=split)
    
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Limit to num_samples if specified (for quick testing)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")
    
    return dataset


# ============================================================================
# PART 2: SFT FORMATTING
# ============================================================================
# For SFT, we concatenate: prompt + chosen response
# Both are treated as training targets (language modeling objective)

def format_prompt_completion_for_sft(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
) -> Dict[str, Any]:
    """
    Format HH-RLHF data for SFT training.
    
    HH-RLHF dataset structure:
    - "chosen": Full conversation string (human + assistant response)
    - "rejected": Full conversation string (human + different assistant response)
    
    For SFT, we use only "chosen" responses.
    
    Args:
        example: Dict with 'chosen' field (full conversation string)
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length
    
    Returns:
        Dict with 'input_ids', 'attention_mask', 'labels'
    """
    
    # HH-RLHF "chosen" is already the full conversation
    # Format: "Human: ...\n\nAssistant: ..."
    text = example["chosen"]
    
    # Tokenize the full conversation
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,           # Truncate if too long
        padding="max_length",      # Pad to max_length
        return_tensors=None,       # Return lists, not tensors
    )
    
    # For causal language modeling, labels = input_ids (predict all tokens)
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": encoding["input_ids"].copy(),
    }


# ============================================================================
# PART 3: DPO FORMATTING
# ============================================================================
# For DPO, we keep prompt, chosen, rejected separate
# DPOTrainer handles the contrastive loss internally

def format_prompt_for_dpo(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_prompt_length: int = 512,
    max_target_length: int = 1024,
) -> Dict[str, Any]:
    """
    Format HH-RLHF data for DPO training.
    
    HH-RLHF structure:
    - "chosen": Full conversation (human + assistant response)
    - "rejected": Full conversation (human + different assistant response)
    
    We extract the human prompt and keep both chosen/rejected responses separate.
    
    DPOTrainer computes: log P(chosen | prompt) vs log P(rejected | prompt)
    And optimizes: log σ(β × (log P(chosen) - log P(rejected)))
    
    Args:
        example: Dict with 'chosen' and 'rejected' fields
        tokenizer: Tokenizer instance
        max_prompt_length: Max prompt length
        max_target_length: Max response length
    
    Returns:
        Dict with 'prompt', 'chosen', 'rejected' (as strings)
    """
    
    # HH-RLHF "chosen" is a full conversation: "Human: ...\n\nAssistant: ..."
    # Extract the prompt (human message) by splitting on "Assistant:"
    chosen_text = example["chosen"]
    rejected_text = example["rejected"]
    
    # Split on "Assistant:" to get prompt and response
    # Format is typically: "Human: <message>\n\nAssistant: <response>"
    if "Assistant:" in chosen_text:
        prompt, _ = chosen_text.split("Assistant:", 1)
        prompt = prompt.strip()
    else:
        # Fallback: use the whole chosen text as prompt
        prompt = chosen_text
    
    # Extract chosen response (everything after "Assistant:")
    if "Assistant:" in chosen_text:
        chosen_response = "Assistant:" + chosen_text.split("Assistant:", 1)[1]
    else:
        chosen_response = chosen_text
    
    # Extract rejected response (everything after "Assistant:")
    if "Assistant:" in rejected_text:
        rejected_response = "Assistant:" + rejected_text.split("Assistant:", 1)[1]
    else:
        rejected_response = rejected_text
    
    # Truncate if needed
    prompt_tokens = tokenizer.encode(prompt, max_length=max_prompt_length, truncation=True)
    prompt_clean = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    
    chosen_tokens = tokenizer.encode(chosen_response, max_length=max_target_length, truncation=True)
    chosen_clean = tokenizer.decode(chosen_tokens, skip_special_tokens=True)
    
    rejected_tokens = tokenizer.encode(rejected_response, max_length=max_target_length, truncation=True)
    rejected_clean = tokenizer.decode(rejected_tokens, skip_special_tokens=True)
    
    return {
        "prompt": prompt_clean,
        "chosen": chosen_clean,
        "rejected": rejected_clean,
    }


# ============================================================================
# PART 4: BATCH PROCESSING
# ============================================================================
# These functions process entire datasets using multi-processing

def prepare_dataset_for_sft(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    num_proc: int = 4,
) -> Dataset:
    """
    Prepare entire dataset for SFT training.
    
    This applies format_prompt_completion_for_sft to all examples in parallel.
    
    Args:
        dataset: HF Dataset object
        tokenizer: Tokenizer instance
        max_length: Max sequence length
        num_proc: Number of CPU cores for parallel processing
    
    Returns:
        Processed Dataset with input_ids, attention_mask, labels
        
    Example:
        >>> dataset = load_hh_rlhf_dataset()
        >>> prepared = prepare_dataset_for_sft(dataset, tokenizer)
        >>> print(len(prepared))
        160800
    """
    logger.info("Preparing dataset for SFT...")
    
    def format_fn(example):
        return format_prompt_completion_for_sft(example, tokenizer, max_length)
    
    processed = dataset.map(
        format_fn,
        num_proc=num_proc,  # Parallel processing on 4 CPU cores
        remove_columns=dataset.column_names,  # Remove original columns
        desc="Formatting for SFT",
    )
    
    logger.info(f"Prepared {len(processed)} SFT examples")
    return processed


def prepare_dataset_for_dpo(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_prompt_length: int = 512,
    max_target_length: int = 1024,
    num_proc: int = 4,
) -> Dataset:
    """
    Prepare entire dataset for DPO training.
    
    This applies format_prompt_for_dpo to all examples in parallel.
    
    Args:
        dataset: HF Dataset object
        tokenizer: Tokenizer instance
        max_prompt_length: Max prompt length
        max_target_length: Max response length
        num_proc: Number of CPU cores for parallel processing
    
    Returns:
        Processed Dataset with prompt, chosen, rejected
        
    Example:
        >>> dataset = load_hh_rlhf_dataset()
        >>> prepared = prepare_dataset_for_dpo(dataset, tokenizer)
        >>> print(prepared[0].keys())
        dict_keys(['prompt', 'chosen', 'rejected'])
    """
    logger.info("Preparing dataset for DPO...")
    
    def format_fn(example):
        return format_prompt_for_dpo(
            example,
            tokenizer,
            max_prompt_length,
            max_target_length,
        )
    
    processed = dataset.map(
        format_fn,
        num_proc=num_proc,  # Parallel processing
        remove_columns=dataset.column_names,
        desc="Formatting for DPO",
    )
    
    logger.info(f"Prepared {len(processed)} DPO examples")
    return processed
