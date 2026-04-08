"""
DPO (Direct Preference Optimization) Training Module

Stage 2 of alignment training: Learn to prefer chosen over rejected responses.

Key Concepts:
- DPO learns preference pairs: (prompt, chosen, rejected)
- Objective: Maximize log P(chosen|prompt) - log P(rejected|prompt)
- Beta controls preference strength (0.1 = good balance)
- Lower learning rate than SFT (5e-5 vs 2e-4)

Reference: Rafailov et al. "Direct Preference Optimization"
https://arxiv.org/abs/2305.18290
"""

import logging
import argparse
from pathlib import Path
from typing import Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType

from ds606.config import TrainingConfig, load_config_from_yaml
from ds606.data.hh_rlhf import load_hh_rlhf_dataset, prepare_dataset_for_dpo

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: MODEL SETUP (Similar to SFT, but loading SFT checkpoint)
# ============================================================================

def setup_model_and_tokenizer_for_dpo(
    model_config,
    sft_model_path: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model & tokenizer for DPO training.
    
    If sft_model_path provided: Load SFT-trained model + LoRA adapter
    Otherwise: Load base model (for comparison, not recommended)
    
    Args:
        model_config: ModelConfig dataclass with model settings
        sft_model_path: Path to SFT checkpoint (e.g., outputs/models/sft/)
    
    Returns:
        (model, tokenizer) tuple
    """
    logger.info("Loading tokenizer from " + model_config.name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_config.name_or_path)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "right"  # Important for training
    tokenizer.truncation_side = "left"
    
    logger.info(f"Loading model from {model_config.name_or_path}")
    
    # Load base model
    torch_dtype = torch.bfloat16 if model_config.torch_dtype == "bfloat16" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.name_or_path,
        torch_dtype=torch_dtype,
        device_map=model_config.device_map,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation="sdpa",
    )
    
    logger.info(f"Model loaded: {model.config.model_type}")
    logger.info(f"Model size: {model.get_memory_footprint() / 1e9:.2f} GB")
    
    # Load SFT adapter if provided
    if sft_model_path:
        logger.info(f"\nLoading SFT adapter from {sft_model_path}")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model,
                sft_model_path,
                is_trainable=True,
            )
            logger.info("✓ SFT adapter loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load SFT adapter: {e}")
            logger.warning("Will train from base model weights instead")
    
    return model, tokenizer


# ============================================================================
# PART 2: LORA SETUP FOR DPO
# ============================================================================

def setup_lora_for_dpo(model, lora_config):
    """
    Apply LoRA to model for DPO training.
    
    Note: If loading from SFT checkpoint, LoRA is already applied.
    This is for cases where we start from base model.
    
    Args:
        model: Model to apply LoRA to
        lora_config: LoraConfig dataclass
    
    Returns:
        LoRA-wrapped model
    """
    if not lora_config.use_lora:
        logger.info("LoRA disabled, using full model fine-tuning")
        return model
    
    logger.info(f"Applying LoRA with r={lora_config.r}, alpha={lora_config.lora_alpha}")
    
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_config.target_modules,
    )
    
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")
    
    return model


# ============================================================================
# PART 3: DPO TRAINING ORCHESTRATOR
# ============================================================================

def train_dpo(
    config: TrainingConfig,
    sft_model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, DPOTrainer]:
    """
    Complete DPO training pipeline.
    
    Args:
        config: TrainingConfig with all hyperparameters
        sft_model_path: Path to SFT checkpoint (recommended)
        output_dir: Override output directory
        resume_from_checkpoint: Resume from checkpoint path
    
    Returns:
        (model, trainer) tuple after training
    """
    
    # Use provided output_dir or from config
    output_dir = output_dir or config.dpo.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    dpo_config = config.dpo
    model_config = config.model
    lora_config = config.lora
    
    # ========== STEP 1: Load Model & Tokenizer ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading Model & Tokenizer")
    logger.info("=" * 80)
    
    model, tokenizer = setup_model_and_tokenizer_for_dpo(
        model_config,
        sft_model_path=sft_model_path,
    )
    
    # ========== STEP 2: Apply LoRA (if starting from base) ==========
    if sft_model_path is None:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Applying LoRA")
        logger.info("=" * 80)
        model = setup_lora_for_dpo(model, lora_config)
    else:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: LoRA already loaded from SFT checkpoint")
        logger.info("=" * 80)
    
    # ========== STEP 3: Load DPO Dataset ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Loading HH-RLHF Dataset for DPO")
    logger.info("=" * 80)
    
    dataset = load_hh_rlhf_dataset(
        split=dpo_config.dataset_split,
        num_samples=5000,  # Same 5K as SFT for consistency
    )
    
    # Split into train/eval
    logger.info(f"Total samples: {len(dataset)}")
    split = dataset.train_test_split(test_size=0.05, seed=dpo_config.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # ========== STEP 4: Prepare Dataset for DPO ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Preparing Dataset for DPO")
    logger.info("=" * 80)
    
    train_dataset = prepare_dataset_for_dpo(
        train_dataset,
        tokenizer,
        max_prompt_length=512,
        max_target_length=1024,
        num_proc=4,
    )
    
    eval_dataset = prepare_dataset_for_dpo(
        eval_dataset,
        tokenizer,
        max_prompt_length=512,
        max_target_length=1024,
        num_proc=4,
    )
    
    # ========== STEP 5: Create DPO Config ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Creating DPO Training Config")
    logger.info("=" * 80)
    
    training_config = DPOConfig(
        output_dir=output_dir,
        
        # Training duration
        num_train_epochs=dpo_config.num_train_epochs,
        max_steps=dpo_config.max_steps if dpo_config.max_steps != -1 else -1,
        
        # Batch sizes
        per_device_train_batch_size=dpo_config.per_device_train_batch_size,
        per_device_eval_batch_size=dpo_config.per_device_eval_batch_size,
        gradient_accumulation_steps=dpo_config.gradient_accumulation_steps,
        
        # Learning rate
        learning_rate=dpo_config.learning_rate,
        lr_scheduler_type=dpo_config.lr_scheduler_type,
        warmup_ratio=dpo_config.warmup_ratio,
        
        # Regularization
        weight_decay=dpo_config.weight_decay,
        max_grad_norm=dpo_config.max_grad_norm,
        
        # Logging & checkpointing
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        
        # Strategies
        save_strategy="steps",
        eval_strategy="steps",
        logging_strategy="steps",
        
        # Precision & optimization
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        
        # DPO-specific
        beta=dpo_config.beta,
        
        # Other
        report_to=[],
        seed=dpo_config.seed,
        dataloader_pin_memory=True,
        optim="adamw_8bit",
    )
    
    logger.info("DPO Config created:")
    logger.info(f"  • Epochs: {training_config.num_train_epochs}")
    logger.info(f"  • Batch size (eff): {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}")
    logger.info(f"  • Learning rate: {training_config.learning_rate}")
    logger.info(f"  • Beta (preference strength): {dpo_config.beta}")
    logger.info(f"  • Saving every {training_config.save_steps} steps")
    
    # ========== STEP 6: Create DPOTrainer ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Creating DPO Trainer")
    logger.info("=" * 80)
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # ========== STEP 7: Start Training ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Starting DPO Training!")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Beta: {dpo_config.beta} (preference strength parameter)")
    logger.info("=" * 80 + "\n")
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # ========== STEP 8: Save Final Model ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Training Complete! Saving Model")
    logger.info("=" * 80)
    
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    
    # Extract training metrics - DPOTrainer returns different format than standard Trainer
    training_loss = getattr(train_result, 'training_loss', None)
    training_runtime = None
    
    # Try different ways to get training_runtime depending on source
    if hasattr(train_result, 'training_runtime'):
        training_runtime = train_result.training_runtime
    elif isinstance(train_result, dict) and 'train_runtime' in train_result:
        training_runtime = train_result['train_runtime']
    elif hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
        # Look for train_runtime in log history
        for log in reversed(trainer.state.log_history):
            if 'train_runtime' in log:
                training_runtime = log['train_runtime']
                break
    
    logger.info(f"Training Results:")
    if training_loss is not None:
        logger.info(f"  • Final training loss: {training_loss:.4f}")
    if training_runtime is not None:
        logger.info(f"  • Training time: {training_runtime / 3600:.2f} hours")
    logger.info("\n✓ DPO training completed successfully!")
    
    return model, trainer


# ============================================================================
# MAIN ENTRY POINT (FOR CLI)
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO training for alignment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_dpo.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--sft-model",
        type=str,
        default=None,
        help="Path to SFT model checkpoint (recommended)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config = load_config_from_yaml(args.config)
    
    # Run training
    train_dpo(
        config,
        sft_model_path=args.sft_model or "outputs/models/sft",
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
