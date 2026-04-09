"""
Evaluation module for comparing base and aligned models.
Generates predictions on both English and Hindi datasets.
"""

import os
import json
import logging
import torch
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

# Setup logging
logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load base model and tokenizer efficiently.
    
    Args:
        model_name: HuggingFace model identifier
        device_map: Device mapping (auto, cuda:0, etc.)
        torch_dtype: Data type for model (bfloat16, float16, float32)
    
    Returns:
        model, tokenizer
    """
    logger.info(f"Loading base model: {model_name}")
    
    # Convert torch_dtype string to actual dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype_obj = dtype_map.get(torch_dtype, torch.bfloat16)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with bfloat16 (no 8-bit quantization to avoid device_map conflicts)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype_obj,
        attn_implementation="sdpa",  # Use SDPA instead of FlashAttention2
    )
    
    model.eval()
    logger.info(f"✓ Base model loaded on {device_map}")
    
    return model, tokenizer


def load_aligned_model(
    base_model: AutoModelForCausalLM,
    adapter_path: str,
) -> AutoModelForCausalLM:
    """
    Load LoRA adapter from SFT/DPO training onto base model.
    
    Args:
        base_model: Base model to attach adapter to
        adapter_path: Path to LoRA adapter directory
    
    Returns:
        Model with adapter loaded
    """
    try:
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False,
        )
        logger.info("✓ LoRA adapter loaded successfully")
        return model
    except Exception as e:
        logger.warning(f"Failed to load adapter from {adapter_path}: {e}")
        logger.warning("Using base model without adapter")
        return base_model


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 1028,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate model response for a given prompt.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate (default: 1028 for comprehensive responses)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
    
    Returns:
        Generated text (response only, without prompt)
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with torch.no_grad() for efficiency
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract response (remove prompt)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = full_response[len(prompt):].strip()
    
    return response_only


def evaluate_models(
    csv_path: str,
    base_model_name: str = "meta-llama/Meta-Llama-3-8B",
    aligned_model_path: str = "outputs/models/dpo/",
    device_map: str = "auto",
    output_path: str = "outputs/evaluations/",
    max_samples: int = None,
    batch_size: int = 1,  # Process one at a time for safety
) -> None:
    """
    Evaluate base and aligned models on English and Hindi prompts.
    
    Args:
        csv_path: Path to CSV with English and hindi columns
        base_model_name: HuggingFace model ID for base model
        aligned_model_path: Path to aligned model (adapter)
        device_map: Device mapping
        output_path: Directory to save results
        max_samples: Maximum samples to evaluate (None = all)
        batch_size: Batch size for generation
    """
    
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 80)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} prompts from {csv_path}")
    
    if max_samples:
        df = df.head(max_samples)
        logger.info(f"Using first {max_samples} samples")
    
    logger.info("=" * 80)
    logger.info("STEP 2: Setting Up Models")
    logger.info("=" * 80)
    
    # Load base model and tokenizer
    base_model, tokenizer = setup_model_and_tokenizer(
        model_name=base_model_name,
        device_map=device_map,
    )
    
    # Load aligned model (with adapter)
    aligned_model = load_aligned_model(base_model, aligned_model_path)
    
    logger.info("=" * 80)
    logger.info("STEP 3: Generating Predictions")
    logger.info("=" * 80)
    
    # Initialize result columns
    results = {
        "base_english": [],
        "base_hindi": [],
        "aligned_english": [],
        "aligned_hindi": [],
    }
    
    total_samples = len(df)
    
    for idx, row in tqdm(df.iterrows(), total=total_samples, desc="Evaluating"):
        english_prompt = row["English"]
        hindi_prompt = row["hindi"]
        
        try:
            # Base model predictions
            logger.debug(f"[{idx+1}/{total_samples}] Generating base model predictions")
            base_en = generate_response(base_model, tokenizer, english_prompt)
            base_hi = generate_response(base_model, tokenizer, hindi_prompt)
            
            # Aligned model predictions
            logger.debug(f"[{idx+1}/{total_samples}] Generating aligned model predictions")
            aligned_en = generate_response(aligned_model, tokenizer, english_prompt)
            aligned_hi = generate_response(aligned_model, tokenizer, hindi_prompt)
            
            results["base_english"].append(base_en)
            results["base_hindi"].append(base_hi)
            results["aligned_english"].append(aligned_en)
            results["aligned_hindi"].append(aligned_hi)
            
        except Exception as e:
            logger.error(f"Error processing sample {idx+1}: {e}")
            # Add placeholder for failed samples
            results["base_english"].append(f"ERROR: {str(e)}")
            results["base_hindi"].append(f"ERROR: {str(e)}")
            results["aligned_english"].append(f"ERROR: {str(e)}")
            results["aligned_hindi"].append(f"ERROR: {str(e)}")
    
    logger.info("=" * 80)
    logger.info("STEP 4: Saving Results")
    logger.info("=" * 80)
    
    # Create results dataframe
    results_df = df.copy()
    for col, values in results.items():
        results_df[col] = values
    
    # Save to CSV
    output_file = os.path.join(output_path, "evaluation_results.csv")
    results_df.to_csv(output_file, index=False)
    logger.info(f"✓ Results saved to {output_file}")
    
    # Save summary statistics
    summary_file = os.path.join(output_path, "evaluation_summary.json")
    summary = {
        "total_samples_evaluated": len(df),
        "base_model": base_model_name,
        "aligned_model_path": aligned_model_path,
        "output_columns": list(results.keys()),
        "csv_path": csv_path,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Summary saved to {summary_file}")
    
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Columns: {list(results.keys())}")


if __name__ == "__main__":
    # Example usage
    evaluate_models(
        csv_path="jailbreak_just_question.csv",
        base_model_name="meta-llama/Meta-Llama-3-8B",
        aligned_model_path="outputs/models/dpo/",
        device_map="auto",
        output_path="outputs/evaluations/",
        max_samples=None,  # Set to small number for testing
    )
