"""
SFT (Supervised Fine-Tuning) Training Module

This module implements the first stage of alignment training:
1. Load base model and tokenizer
2. Apply LoRA for memory efficiency
3. Train on HH-RLHF dataset using SFT objective
4. Save checkpoints and final model

Training Flow:
  Model + Tokenizer
      ↓
    LoRA (4.2M params)
      ↓
    Dataset (SFT format)
      ↓
    Trainer
      ↓
    Training Loop
      ↓
    Checkpoints + Final Model
"""

from __future__ import annotations

import logging
import torch
from pathlib import Path
from typing import Optional
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig as PeftLoraConfig, TaskType
from datasets import load_dataset

from ds606.config import TrainingConfig, load_config_from_yaml
from ds606.data.hh_rlhf import (
    load_hh_rlhf_dataset,
    prepare_dataset_for_sft,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# PART 1: LOAD MODEL AND TOKENIZER
# ============================================================================
# This part:
# 1. Downloads model from HuggingFace Hub
# 2. Loads tokenizer
# 3. Handles precision (bfloat16) and device placement (auto)

def setup_model_and_tokenizer(
    model_config,
    use_flash_attention_2: bool = True,
) -> tuple:
    """
    Load and setup base model and tokenizer.
    
    This function:
    - Downloads model from HF Hub (if not cached)
    - Loads tokenizer with proper settings
    - Configures dtype and device mapping
    - Optionally enables Flash Attention 2 for speed
    
    Args:
        model_config: ModelConfig object with model settings
        use_flash_attention_2: Enable Flash Attention 2? (2x faster)
    
    Returns:
        (model, tokenizer) tuple
        
    Key settings:
      - torch_dtype: bfloat16 (16-bit, faster & less memory)
      - device_map: "auto" or "cuda:0" (auto-distribute across GPUs)
      - Flash Attention 2: 2x faster attention computation
      
    Example:
        >>> model, tokenizer = setup_model_and_tokenizer(config.model)
        >>> print(model)
        LlamaForCausalLM(...)
    """
    
    # ========== Step 1: Load Tokenizer ==========
    logger.info(f"Loading tokenizer from {model_config.name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        padding_side="right",  # Important: Add padding on right (not left)
                               # This is crucial for training!
    )
    
    # ========== Step 2: Handle Missing Padding Token ==========
    # Some models don't have a padding token defined
    # We use EOS (end-of-sequence) token as padding
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # ========== Step 3: Load Model ==========
    logger.info(f"Loading model from {model_config.name_or_path}")
    
    # Convert dtype string to torch dtype
    # "bfloat16" → torch.bfloat16 (faster, less memory)
    # "float32" → torch.float32 (more memory, more precise)
    torch_dtype = torch.bfloat16 if model_config.torch_dtype == "bfloat16" else torch.float16
    
    # Load model with proper settings
    model = AutoModelForCausalLM.from_pretrained(
        model_config.name_or_path,
        torch_dtype=torch_dtype,
        device_map=model_config.device_map,  # "auto", "cuda:0", etc.
        trust_remote_code=model_config.trust_remote_code,
        # Use SDPA (Scaled Dot Product Attention) - efficient & no extra dependencies needed
        attn_implementation="sdpa",
    )
    
    logger.info(f"Model loaded: {model.config.model_type}")
    logger.info(f"Model size: {model.get_memory_footprint() / 1e9:.2f} GB")
    
    return model, tokenizer


# ============================================================================
# PART 2: APPLY LORA (LOW-RANK ADAPTATION)
# ============================================================================
# This part:
# 1. Creates LoRA configuration
# 2. Wraps model with LoRA adapters
# 3. Freezes base model, only trains 4.2M LoRA parameters

def setup_lora(
    model: AutoModelForCausalLM,
    lora_config,
) -> AutoModelForCausalLM:
    """
    Apply LoRA (Low-Rank Adaptation) to model.
    
    Why LoRA?
      Base model:    8B parameters (80GB memory to train)
      ➜ With LoRA:   4.2M parameters (40GB memory)
      Same performance, half the memory! 💾
    
    How LoRA works:
      1. Freeze all base model weights
      2. Add small "adapter" layers (rank r=16)
      3. Only train the adapters (4.2M params)
      4. Combine during inference: output = base + adapter
    
    Args:
        model: AutoModelForCausalLM to apply LoRA to
        lora_config: LoraConfig object with settings
    
    Returns:
        Model wrapped with LoRA adapters
        
    Example:
        >>> model = setup_lora(model, config.lora)
        >>> model.print_trainable_parameters()
        trainable params: 4,194,304 || all params: 8,030,261,248
    """
    
    if not lora_config.use_lora:
        logger.info("LoRA disabled - training all 8B parameters (warning: uses 80GB+)")
        return model
    
    logger.info(f"Applying LoRA with r={lora_config.r}, alpha={lora_config.lora_alpha}")
    
    # ========== Create LoRA Configuration ==========
    peft_config = PeftLoraConfig(
        # LoRA rank: amount of variance we can represent
        # Higher r = more expressive but more parameters
        r=lora_config.r,  # 16 is good default
        
        # Scaling factor: controls how much LoRA affects output
        # Usually 2x the rank
        lora_alpha=lora_config.lora_alpha,  # 32
        
        # Dropout in LoRA layers (regularization)
        lora_dropout=lora_config.lora_dropout,  # 0.05
        
        # Which parts of model to apply LoRA to
        # q_proj, v_proj = parts of attention mechanism
        target_modules=lora_config.target_modules,  # ["q_proj", "v_proj"]
        
        # Don't train bias terms
        bias=lora_config.bias,  # "none"
        
        # Task type: causal language modeling (next token prediction)
        task_type=TaskType.CAUSAL_LM,
    )
    
    # ========== Wrap Model with LoRA ==========
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameter info
    model.print_trainable_parameters()
    # Output: trainable params: 4,194,304 || all params: 8,030,261,248
    
    return model


# ============================================================================
# PART 3: MAIN SFT TRAINING FUNCTION
# ============================================================================
# This is the orchestrator that combines everything

def train_sft(
    config: TrainingConfig,
    output_dir: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> tuple:
    """
    Main SFT training function.
    
    This function orchestrates the entire SFT training pipeline:
    1. Load model and tokenizer
    2. Apply LoRA
    3. Load and prepare dataset
    4. Create trainer with hyperparameters
    5. Run training loop
    6. Save final model
    
    Args:
        config: TrainingConfig object with all settings
        output_dir: Override output directory
        resume_from_checkpoint: Path to checkpoint to resume from
    
    Returns:
        (trained_model, trainer) tuple
        
    Example:
        >>> config = TrainingConfig()
        >>> model, trainer = train_sft(config)
        >>> # Training runs here...
    """
    
    sft_config = config.sft
    output_dir = output_dir or sft_config.output_dir
    
    # ========== Step 1: Setup Model & Tokenizer ==========
    logger.info("=" * 80)
    logger.info("STEP 1: Loading Model & Tokenizer")
    logger.info("=" * 80)
    
    model, tokenizer = setup_model_and_tokenizer(
        config.model,
        use_flash_attention_2=config.use_flash_attention_2,
    )
    
    # ========== Step 2: Apply LoRA ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Applying LoRA")
    logger.info("=" * 80)
    
    model = setup_lora(model, config.lora)
    
    # ========== Step 3: Load Dataset ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Loading HH-RLHF Dataset")
    logger.info("=" * 80)
    
    dataset = load_hh_rlhf_dataset(
        split=sft_config.dataset_split,
        num_samples=None,  # Use all samples
    )
    
    # ========== Step 4: Split into Train/Eval ==========
    logger.info(f"Total samples: {len(dataset)}")
    
    # Create 95-5 split for train-eval
    split = dataset.train_test_split(test_size=0.05, seed=sft_config.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # ========== Step 5: Prepare Datasets for Training ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Preparing Dataset for SFT")
    logger.info("=" * 80)
    
    train_dataset = prepare_dataset_for_sft(
        train_dataset,
        tokenizer,
        max_length=sft_config.max_seq_length,
        num_proc=sft_config.preprocessing_num_workers,
    )
    
    eval_dataset = prepare_dataset_for_sft(
        eval_dataset,
        tokenizer,
        max_length=sft_config.max_seq_length,
        num_proc=sft_config.preprocessing_num_workers,
    )
    
    # ========== Step 6: Create Training Arguments ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Creating Training Arguments")
    logger.info("=" * 80)
    
    training_args = TrainingArguments(
        # Output and evaluation
        output_dir=output_dir,
        
        # Training duration
        num_train_epochs=sft_config.num_train_epochs,
        max_steps=sft_config.max_steps,
        
        # Batch sizes
        per_device_train_batch_size=sft_config.per_device_train_batch_size,
        per_device_eval_batch_size=sft_config.per_device_eval_batch_size,
        gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
        
        # Learning rate schedule
        learning_rate=sft_config.learning_rate,
        lr_scheduler_type=sft_config.lr_scheduler_type,
        warmup_ratio=sft_config.warmup_ratio,  # Warmup first 10% of training
        
        # Regularization
        weight_decay=sft_config.weight_decay,
        max_grad_norm=sft_config.max_grad_norm,
        
        # Logging and saving
        logging_steps=sft_config.logging_steps,
        save_steps=sft_config.save_steps,
        eval_steps=sft_config.eval_steps,
        save_total_limit=sft_config.save_total_limit,
        
        # Strategies
        save_strategy="steps",
        eval_strategy="steps",  # Changed from evaluation_strategy for newer transformers
        logging_strategy="steps",
        
        # Precision and optimization
        bf16=True,  # Use bfloat16 for training (faster, less memory)
        tf32=True,  # Allow tf32 for speedup on newer GPUs
        gradient_checkpointing=True,  # Save memory during backprop
        
        # Other
        report_to=sft_config.report_to,  # ["wandb"] = log to Weights & Biases
        seed=sft_config.seed,
        dataloader_pin_memory=True,  # Faster data loading
        optim="adamw_8bit",  # 8-bit optimizer (memory efficient)
    )
    
    logger.info("Training Arguments created:")
    logger.info(f"  • Epochs: {training_args.num_train_epochs}")
    logger.info(f"  • Batch size (eff): {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  • Learning rate: {training_args.learning_rate}")
    logger.info(f"  • Saving every {training_args.save_steps} steps")
    
    # ========== Step 7: Create Trainer ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Creating Trainer")
    logger.info("=" * 80)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # ========== Step 8: Run Training ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Starting Training!")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info("Monitor progress with: wandb offline or visit wandb.ai")
    logger.info("=" * 80 + "\n")
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # ========== Step 9: Save Final Model ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Training Complete! Saving Model")
    logger.info("=" * 80)
    
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    
    logger.info(f"Training Results:")
    logger.info(f"  • Final training loss: {train_result.training_loss:.4f}")
    logger.info(f"  • Training time: {train_result.training_time_hrs:.2f} hours")
    
    return model, trainer


# ============================================================================
# PART 4: MAIN ENTRY POINT (FOR CLI)
# ============================================================================
# This allows running as: python -m ds606.models.sft --config configs/training_sft.yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT training for alignment")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_sft.yaml",
        help="Path to training config YAML",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )
    
    args = parser.parse_args()
    
    # Load config from YAML file
    logger.info(f"Loading config from {args.config}")
    
    if Path(args.config).exists():
        config = load_config_from_yaml(args.config)
    else:
        logger.warning(f"Config file {args.config} not found, using defaults")
        config = TrainingConfig()
    
    # Run training
    train_sft(
        config,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
