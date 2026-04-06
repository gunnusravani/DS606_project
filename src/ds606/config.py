from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml
from pathlib import Path


# ============================================================================
# PART 1: MODEL CONFIGURATION
# ============================================================================
# This defines settings for loading the base model
# Think of it as: "What model do we start with and how do we load it?"

@dataclass
class ModelConfig:
    """Configuration for loading the base model."""
    name_or_path: str = "meta-llama/Meta-Llama-3-8B"
    # ^ Which model to download from HuggingFace Hub
    
    torch_dtype: str = "bfloat16"
    # ^ Data type: bfloat16 is 16-bit (faster, less memory) vs float32 (slower, more memory)
    
    device_map: str = "cuda:0"
    # ^ "auto" = automatically split model across available GPUs/CPUs
    
    trust_remote_code: bool = True
    # ^ Allow downloading custom code from HF Hub (some models need this)


# ============================================================================
# PART 2: LORA CONFIGURATION
# ============================================================================
# LoRA = Low-Rank Adaptation
# Problem: Fine-tuning 8B params needs 80GB memory
# Solution: Only train 4.2M params using LoRA "adapters"
# How: Add small trainable layers (rank r=16) + freeze everything else

@dataclass
class LoraConfig:
    """Configuration for LoRA (Low-Rank Adapters)."""
    
    use_lora: bool = True
    # ^ Enable LoRA? (True = memory efficient, False = full fine-tuning)
    
    r: int = 16
    # ^ Rank of LoRA adapter. Higher = more parameters but slower
    # Think of it as: "How wide is the adapter matrix?"
    
    lora_alpha: int = 32
    # ^ Scaling factor. Usually 2x the rank
    # Controls how much the LoRA layer affects the model
    
    lora_dropout: float = 0.05
    # ^ Dropout in LoRA layers (prevents overfitting)
    
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    # ^ Which parts of the transformer to apply LoRA to
    # q_proj = query projection, v_proj = value projection (in attention layers)
    
    bias: str = "none"
    # ^ Don't add bias terms (simpler, works well)
    
    task_type: str = "CAUSAL_LM"
    # ^ Causal Language Modeling = predicting next token (what LLMs do)


# ============================================================================
# PART 3: SFT (SUPERVISED FINE-TUNING) CONFIGURATION
# ============================================================================
# SFT = Learn from high-quality human-written responses
# Example: Prompt = "How do I...?" → Response = "Here's how you can safely..."

@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning."""
    
    # Data settings
    dataset_name: str = "Anthropic/hh-rlhf"
    # ^ The dataset to train on (160K examples)
    
    dataset_split: str = "train"
    # ^ Use the "train" split (not test/validation)
    
    # Training hyperparameters
    num_train_epochs: int = 3
    # ^ How many times to go through the dataset (3 = optimal for SFT)
    
    max_steps: int = -1
    # ^ Max training steps. -1 = go until all epochs done
    
    per_device_train_batch_size: int = 4
    # ^ How many examples to train on at once per GPU
    # Smaller batch = less memory but noisier training
    
    per_device_eval_batch_size: int = 8
    # ^ Batch size for evaluation (can be larger, no backprop)
    
    gradient_accumulation_steps: int = 2
    # ^ Accumulate gradients over 2 batches before updating weights
    # Effect: Same as batch size 8, but using only 40GB memory
    
    learning_rate: float = 2e-4
    # ^ How big are weight updates? 2e-4 = 0.0002 (small, careful updates)
    
    lr_scheduler_type: str = "cosine"
    # ^ Learning rate schedule: start high, slowly decrease to 0 (warmup + cosine decay)
    
    warmup_ratio: float = 0.1
    # ^ First 10% of training: gradually increase LR from 0 to peak
    
    weight_decay: float = 0.01
    # ^ L2 regularization (penalize large weights, prevents overfitting)
    
    max_grad_norm: float = 1.0
    # ^ Gradient clipping (if gradient norm > 1, scale it down)
    
    seed: int = 42
    # ^ Random seed for reproducibility
    
    # Logging & checkpointing
    logging_steps: int = 100
    # ^ Print/log loss every 100 steps
    
    save_steps: int = 500
    # ^ Save model checkpoint every 500 steps
    
    eval_steps: int = 500
    # ^ Evaluate on validation set every 500 steps
    
    save_total_limit: int = 3
    # ^ Keep only last 3 checkpoints (save disk space)
    
    # Data processing
    max_seq_length: int = 2048
    # ^ Max length of input sequence, pad/truncate to this
    
    preprocessing_num_workers: int = 4
    # ^ Use 4 CPU cores to preprocess data in parallel
    
    overwrite_cache: bool = False
    # ^ Don't recompute cached processed data
    
    push_to_hub: bool = False
    # ^ Don't automatically upload final model to HF Hub
    
    # Output
    output_dir: str = "outputs/models/sft"
    # ^ Where to save model checkpoints
    
    report_to: list[str] = field(default_factory=lambda: ["wandb"])
    # ^ Report metrics to Weights & Biases (wandb)


# ============================================================================
# PART 4: DPO (DIRECT PREFERENCE OPTIMIZATION) CONFIGURATION
# ============================================================================
# DPO = Learn to prefer chosen responses over rejected ones
# Example: Prompt = "How do I...?"
#          Chosen = "Here's how you can safely..."
#          Rejected = "I can't help with that..."
#          Goal: Make model output (Chosen) > (Rejected)

@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization."""
    
    # Data settings
    dataset_name: str = "Anthropic/hh-rlhf"
    # ^ Same dataset as SFT
    
    dataset_split: str = "train"
    
    # Training hyperparameters
    num_train_epochs: int = 2
    # ^ DPO typically needs fewer epochs than SFT
    
    max_steps: int = -1
    
    per_device_train_batch_size: int = 2
    # ^ Smaller batch than SFT because DPO pairs are heavier (2 responses per prompt)
    
    per_device_eval_batch_size: int = 4
    
    gradient_accumulation_steps: int = 4
    # ^ Accumulate over 4 batches (batch 2 * accumulation 4 = effective batch 8)
    
    learning_rate: float = 5e-5
    # ^ Much lower than SFT (0.00005)! DPO is more sensitive to LR
    
    lr_scheduler_type: str = "cosine"
    
    warmup_ratio: float = 0.1
    
    weight_decay: float = 0.01
    
    max_grad_norm: float = 1.0
    
    seed: int = 42
    
    # DPO-specific hyperparameters
    beta: float = 0.1
    # ^ Temperature parameter: controls preference strength
    # Higher beta = sharper preference signal (model learns faster but less stable)
    
    label_smoothing: float = 0.0
    # ^ Smooth the preference labels (0.0 = no smoothing)
    
    loss_type: str = "sigmoid"
    # ^ DPO loss function: "sigmoid", "hinge", "ipo"
    
    # Logging & checkpointing
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Data processing
    max_prompt_length: int = 512
    # ^ Truncate long prompts
    
    max_target_length: int = 1024
    # ^ Truncate long responses
    
    preprocessing_num_workers: int = 4
    
    # Output
    output_dir: str = "outputs/models/dpo"
    report_to: list[str] = field(default_factory=lambda: ["wandb"])


# ============================================================================
# PART 5: MASTER TRAINING CONFIG
# ============================================================================
# This combines all the sub-configs into one master config

@dataclass
class TrainingConfig:
    """Master configuration combining all training settings."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    # ^ Model loading settings
    
    lora: LoraConfig = field(default_factory=LoraConfig)
    # ^ LoRA adapter settings
    
    sft: SFTConfig = field(default_factory=SFTConfig)
    # ^ SFT training hyperparameters
    
    dpo: DPOConfig = field(default_factory=DPOConfig)
    # ^ DPO training hyperparameters
    
    stage: str = "sft"
    # ^ Which stage to run: "sft" or "dpo"
    
    use_flash_attention_2: bool = True
    # ^ Use Flash Attention 2 (2x faster attention, uses less memory)


# ============================================================================
# PART 6: YAML LOADER FUNCTION
# ============================================================================
# This lets us load config from a YAML file instead of hardcoding in Python

def load_config_from_yaml(config_path: str) -> TrainingConfig:
    """
    Load training config from a YAML file.
    
    Args:
        config_path: Path to YAML file (e.g., "configs/training_sft.yaml")
    
    Returns:
        TrainingConfig object with all settings
        
    Example:
        config = load_config_from_yaml("configs/training_sft.yaml")
        print(config.sft.learning_rate)  # 0.0002
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create each sub-config from YAML values
    model_cfg = ModelConfig(**config_dict.get('model', {}))
    lora_cfg = LoraConfig(**config_dict.get('lora', {}))
    sft_cfg = SFTConfig(**config_dict.get('sft', {}))
    dpo_cfg = DPOConfig(**config_dict.get('dpo', {}))
    
    # Combine into master config
    return TrainingConfig(
        model=model_cfg,
        lora=lora_cfg,
        sft=sft_cfg,
        dpo=dpo_cfg,
        stage=config_dict.get('stage', 'sft'),
        use_flash_attention_2=config_dict.get('use_flash_attention_2', True),
    )
