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
from dotenv import load_dotenv
from huggingface_hub import login

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

# Load environment variables from .env file
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

# Authenticate with HuggingFace if HF_TOKEN is available
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    logger.info("✓ Authenticated with HuggingFace using token from .env")


def setup_model_and_tokenizer(
    model_name: str = "meta-llama/Llama-3.1-8B",
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
    base_model_name: str = "meta-llama/Llama-3.1-8B",
    aligned_model_path: str = "models/dpo/",
    device_map: str = "auto",
    output_path: str = "outputs/evaluations/",
    max_samples: int = None,
    batch_size: int = 1,  # Process one at a time for safety
    resume_from_saved: bool = True,  # Fill missing values from existing results
) -> None:
    """
    Evaluate base and aligned models on English and Hindi prompts.
    Supports resuming from incomplete evaluations to fill missing values.
    
    Args:
        csv_path: Path to CSV with English and hindi columns
        base_model_name: HuggingFace model ID for base model
        aligned_model_path: Path to aligned model (adapter)
        device_map: Device mapping
        output_path: Directory to save results
        max_samples: Maximum samples to evaluate (None = all)
        batch_size: Batch size for generation
        resume_from_saved: If True, detect and fill missing values from existing results
    """
    
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 80)
    
    # Load input CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} prompts from {csv_path}")
    
    if max_samples:
        df = df.head(max_samples)
        logger.info(f"Using first {max_samples} samples")
    
    # Check if previous results exist
    output_file = os.path.join(output_path, "evaluation_results.csv")
    prediction_cols = ["base_english", "base_hindi", "aligned_english", "aligned_hindi"]
    
    results_df = None
    rows_to_evaluate = list(range(len(df)))
    
    if resume_from_saved and os.path.exists(output_file):
        logger.info(f"Found existing results at {output_file}")
        results_df = pd.read_csv(output_file)
        logger.info(f"Loaded {len(results_df)} rows from existing results")
        
        # Identify rows with missing predictions
        missing_rows = []
        for col in prediction_cols:
            if col not in results_df.columns:
                missing_rows.extend(list(range(len(results_df))))
                logger.info(f"Missing column: {col}")
                break
            else:
                # Find rows where this column is empty/null/ERROR
                missing = results_df[
                    (results_df[col].isna()) | 
                    (results_df[col] == "") | 
                    (results_df[col].str.startswith("ERROR", na=False))
                ].index.tolist()
                missing_rows.extend(missing)
        
        # Deduplicate and sort
        rows_to_evaluate = sorted(list(set(missing_rows)))
        logger.info(f"Found {len(rows_to_evaluate)} rows with missing/empty predictions")
        
        if len(rows_to_evaluate) == 0:
            logger.info("✓ All rows already have valid predictions!")
            logger.info("=" * 80)
            logger.info("EVALUATION COMPLETE (No updates needed)")
            logger.info("=" * 80)
            return
    else:
        if resume_from_saved:
            logger.info(f"No existing results found. Starting fresh evaluation.")
        # Initialize results_df as copy of input
        results_df = df.copy()
        for col in prediction_cols:
            results_df[col] = ""
    
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
    logger.info("STEP 3: Generating Predictions for Missing Rows")
    logger.info("=" * 80)
    
    total_rows = len(rows_to_evaluate)
    logger.info(f"Evaluating {total_rows} rows...")
    
    for batch_idx, row_idx in tqdm(enumerate(rows_to_evaluate), total=total_rows, desc="Filling missing"):
        try:
            # Get the row data
            row = df.iloc[row_idx]
            english_prompt = row["English"]
            hindi_prompt = row["hindi"]
            
            # Base model predictions
            logger.debug(f"[{batch_idx+1}/{total_rows}] Generating base model predictions for row {row_idx}")
            base_en = generate_response(base_model, tokenizer, english_prompt)
            base_hi = generate_response(base_model, tokenizer, hindi_prompt)
            
            # Aligned model predictions
            logger.debug(f"[{batch_idx+1}/{total_rows}] Generating aligned model predictions for row {row_idx}")
            aligned_en = generate_response(aligned_model, tokenizer, english_prompt)
            aligned_hi = generate_response(aligned_model, tokenizer, hindi_prompt)
            
            # Update the results dataframe
            results_df.at[row_idx, "base_english"] = base_en
            results_df.at[row_idx, "base_hindi"] = base_hi
            results_df.at[row_idx, "aligned_english"] = aligned_en
            results_df.at[row_idx, "aligned_hindi"] = aligned_hi
            
        except Exception as e:
            logger.error(f"Error processing row {row_idx}: {e}")
            # Add placeholder for failed samples
            results_df.at[row_idx, "base_english"] = f"ERROR: {str(e)}"
            results_df.at[row_idx, "base_hindi"] = f"ERROR: {str(e)}"
            results_df.at[row_idx, "aligned_english"] = f"ERROR: {str(e)}"
            results_df.at[row_idx, "aligned_hindi"] = f"ERROR: {str(e)}"
    
    logger.info("=" * 80)
    logger.info("STEP 4: Saving Results")
    logger.info("=" * 80)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    logger.info(f"✓ Results saved to {output_file}")
    
    # Save summary statistics
    summary_file = os.path.join(output_path, "evaluation_summary.json")
    summary = {
        "total_samples_in_csv": len(df),
        "rows_evaluated_in_this_run": len(rows_to_evaluate),
        "base_model": base_model_name,
        "aligned_model_path": aligned_model_path,
        "output_columns": prediction_cols,
        "csv_path": csv_path,
        "resume_enabled": resume_from_saved,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Summary saved to {summary_file}")
    
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Prediction columns: {prediction_cols}")


def evaluate_models_with_initial_response(
    csv_path: str,
    base_model_name: str = "meta-llama/Llama-3.1-8B",
    aligned_model_path: str = "models/dpo/",
    device_map: str = "auto",
    output_path: str = "outputs/evaluations_malicious/",
    max_samples: int = None,
    resume_from_saved: bool = True,
    english_prompt_col: str = "question",
    english_initial_col: str = "intital_malicious_english",
    hindi_prompt_col: str = "hindi",
    hindi_initial_col: str = "intital_malicious_hindi",
) -> None:
    """
    Evaluate models on dataset where prompt is combined with initial malicious response.
    Combines: prompt + initial_response for both English and Hindi.
    
    Args:
        csv_path: Path to CSV with question, hindi, intital_malicious_english, intital_malicious_hindi
        base_model_name: HuggingFace model ID for base model
        aligned_model_path: Path to aligned model (adapter)
        device_map: Device mapping
        output_path: Directory to save results
        max_samples: Maximum samples to evaluate (None = all)
        resume_from_saved: If True, detect and fill missing values from existing results
        english_prompt_col: Column name for English question
        english_initial_col: Column name for English initial response
        hindi_prompt_col: Column name for Hindi question
        hindi_initial_col: Column name for Hindi initial response
    """
    
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("EVALUATION WITH INITIAL MALICIOUS RESPONSES")
    logger.info("=" * 80)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 80)
    
    # Load input CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples from {csv_path}")
    
    if max_samples:
        df = df.head(max_samples)
        logger.info(f"Using first {max_samples} samples")
    
    # Verify required columns exist
    required_cols = [english_prompt_col, english_initial_col, hindi_prompt_col, hindi_initial_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")
    
    logger.info(f"Using columns:")
    logger.info(f"  English: {english_prompt_col} + {english_initial_col}")
    logger.info(f"  Hindi: {hindi_prompt_col} + {hindi_initial_col}")
    
    # Combine prompts with initial responses
    logger.info("Combining prompts with initial malicious responses...")
    df['combined_english'] = df[english_prompt_col].astype(str) + " " + df[english_initial_col].astype(str)
    df['combined_hindi'] = df[hindi_prompt_col].astype(str) + " " + df[hindi_initial_col].astype(str)
    
    logger.info(f"Sample English prompt:\n  {df['combined_english'].iloc[0][:200]}...\n")
    logger.info(f"Sample Hindi prompt:\n  {df['combined_hindi'].iloc[0][:200]}...\n")
    
    # Check if previous results exist
    output_file = os.path.join(output_path, "malicious_initial_results.csv")
    prediction_cols = ["base_english", "base_hindi", "aligned_english", "aligned_hindi"]
    
    results_df = None
    rows_to_evaluate = list(range(len(df)))
    
    if resume_from_saved and os.path.exists(output_file):
        logger.info(f"Found existing results at {output_file}")
        results_df = pd.read_csv(output_file)
        logger.info(f"Loaded {len(results_df)} rows from existing results")
        
        # Identify rows with missing predictions
        missing_rows = []
        for col in prediction_cols:
            if col not in results_df.columns:
                missing_rows.extend(list(range(len(results_df))))
                logger.info(f"Missing column: {col}")
                break
            else:
                # Find rows where this column is empty/null/ERROR
                missing = results_df[
                    (results_df[col].isna()) | 
                    (results_df[col] == "") | 
                    (results_df[col].str.startswith("ERROR", na=False))
                ].index.tolist()
                missing_rows.extend(missing)
        
        # Deduplicate and sort
        rows_to_evaluate = sorted(list(set(missing_rows)))
        logger.info(f"Found {len(rows_to_evaluate)} rows with missing/empty predictions")
        
        if len(rows_to_evaluate) == 0:
            logger.info("✓ All rows already have valid predictions!")
            logger.info("=" * 80)
            logger.info("EVALUATION COMPLETE (No updates needed)")
            logger.info("=" * 80)
            return
    else:
        if resume_from_saved:
            logger.info("No existing results found. Starting fresh evaluation.")
        # Initialize results_df
        results_df = df[[english_prompt_col, hindi_prompt_col, english_initial_col, hindi_initial_col]].copy()
        for col in prediction_cols:
            results_df[col] = ""
    
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
    logger.info("STEP 3: Generating Predictions for Missing Rows")
    logger.info("=" * 80)
    
    total_rows = len(rows_to_evaluate)
    logger.info(f"Evaluating {total_rows} rows...")
    
    for batch_idx, row_idx in tqdm(enumerate(rows_to_evaluate), total=total_rows, desc="Evaluating"):
        try:
            # Get combined prompts
            english_prompt = df.iloc[row_idx]['combined_english']
            hindi_prompt = df.iloc[row_idx]['combined_hindi']
            
            # Base model predictions
            base_en = generate_response(base_model, tokenizer, english_prompt)
            base_hi = generate_response(base_model, tokenizer, hindi_prompt)
            
            # Aligned model predictions
            aligned_en = generate_response(aligned_model, tokenizer, english_prompt)
            aligned_hi = generate_response(aligned_model, tokenizer, hindi_prompt)
            
            # Update the results dataframe
            results_df.at[row_idx, "base_english"] = base_en
            results_df.at[row_idx, "base_hindi"] = base_hi
            results_df.at[row_idx, "aligned_english"] = aligned_en
            results_df.at[row_idx, "aligned_hindi"] = aligned_hi
            
        except Exception as e:
            logger.error(f"Error processing row {row_idx}: {e}")
            results_df.at[row_idx, "base_english"] = f"ERROR: {str(e)}"
            results_df.at[row_idx, "base_hindi"] = f"ERROR: {str(e)}"
            results_df.at[row_idx, "aligned_english"] = f"ERROR: {str(e)}"
            results_df.at[row_idx, "aligned_hindi"] = f"ERROR: {str(e)}"
    
    logger.info("=" * 80)
    logger.info("STEP 4: Saving Results")
    logger.info("=" * 80)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    logger.info(f"✓ Results saved to {output_file}")
    
    # Save summary statistics
    summary_file = os.path.join(output_path, "malicious_initial_summary.json")
    summary = {
        "total_samples": len(df),
        "rows_evaluated_in_this_run": len(rows_to_evaluate),
        "base_model": base_model_name,
        "aligned_model_path": aligned_model_path,
        "output_columns": prediction_cols,
        "csv_path": csv_path,
        "column_mapping": {
            "english_prompt": english_prompt_col,
            "english_initial_response": english_initial_col,
            "hindi_prompt": hindi_prompt_col,
            "hindi_initial_response": hindi_initial_col,
        },
        "resume_enabled": resume_from_saved,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Summary saved to {summary_file}")
    
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Prediction columns: {prediction_cols}")


if __name__ == "__main__":
    # Example usage
    evaluate_models(
        csv_path="jailbreak_just_question.csv",
        base_model_name="meta-llama/Llama-3.1-8B",
        aligned_model_path="models/dpo/",
        device_map="auto",
        output_path="outputs/evaluations/",
        max_samples=None,  # Set to small number for testing
    )
