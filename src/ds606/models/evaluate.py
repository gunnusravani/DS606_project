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
    max_new_tokens: int = 512,  # Increased since we're truncating prompt
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    no_repeat_ngram_size: int = 3,
    repetition_penalty: float = 1.2,
) -> str:
    """
    Generate model response for a given prompt with repetition control.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt (will be truncated to 5000 tokens)
        max_new_tokens: Maximum tokens to generate (default: 512)
        temperature: Sampling temperature (0.7 = good balance of randomness vs coherence)
        top_p: Nucleus sampling parameter (0.95 = keep top 95% probability mass)
        top_k: Top-K filtering (50 = filter to top 50 tokens)
        no_repeat_ngram_size: Prevent n-grams of this size from repeating (3 = no 3-grams repeat)
        repetition_penalty: Penalty for repeating tokens (1.2 = 20% penalty for repeats)
    
    Returns:
        Generated text (response only, without prompt)
    """
    try:
        # Validate prompt
        if not prompt or prompt.strip() == "":
            return "ERROR: Empty prompt"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids_len = inputs['input_ids'].shape[1]
        
        # Truncate prompt to 5000 tokens to leave room for generation
        # 8000 total - 5000 prompt = 3000 tokens available for generation
        max_prompt_tokens = 5000
        if input_ids_len > max_prompt_tokens:
            logger.debug(f"Truncating prompt from {input_ids_len} to {max_prompt_tokens} tokens")
            inputs['input_ids'] = inputs['input_ids'][:, :max_prompt_tokens]
            inputs['attention_mask'] = inputs['attention_mask'][:, :max_prompt_tokens]
            input_ids_len = max_prompt_tokens
        
        # Calculate safe max_new_tokens
        context_limit = 8000  # Llama 3 approximate context
        safe_max_new_tokens = min(max_new_tokens, context_limit - input_ids_len - 50)
        
        if safe_max_new_tokens < 50:
            return f"ERROR: Insufficient space for generation (input_len={input_ids_len})"
        
        # Generate with torch.no_grad() for efficiency
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=safe_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Extract only the generated part (after prompt)
        response_tokens = outputs[0][input_ids_len:]
        if len(response_tokens) > 0:
            response_only = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        else:
            response_only = ""
        
        # If no response generated, try full decoding fallback
        if not response_only or response_only.strip() == "":
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if full_response.startswith(prompt):
                response_only = full_response[len(prompt):].strip()
            else:
                response_only = full_response.strip()
        
        # Final check
        if not response_only or response_only.strip() == "":
            return f"ERROR: No tokens generated (input_len={input_ids_len}, max_new={safe_max_new_tokens})"
        
        return response_only
        
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)[:80]}"


def evaluate_models(
    csv_path: str,
    base_model_name: str = "meta-llama/Meta-Llama-3-8B",
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
    base_model_name: str = "meta-llama/Meta-Llama-3-8B",
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
    logger.info("STEP 2: Creating Combined Prompts")
    logger.info("=" * 80)
    
    # Check for NaN values in required columns
    logger.info(f"Checking for NaN values...")
    logger.info(f"  {english_prompt_col}: {df[english_prompt_col].isna().sum()} NaN")
    logger.info(f"  {english_initial_col}: {df[english_initial_col].isna().sum()} NaN")
    logger.info(f"  {hindi_prompt_col}: {df[hindi_prompt_col].isna().sum()} NaN")
    logger.info(f"  {hindi_initial_col}: {df[hindi_initial_col].isna().sum()} NaN")
    
    # Replace NaN with empty strings
    df[english_prompt_col] = df[english_prompt_col].fillna("")
    df[english_initial_col] = df[english_initial_col].fillna("")
    df[hindi_prompt_col] = df[hindi_prompt_col].fillna("")
    df[hindi_initial_col] = df[hindi_initial_col].fillna("")
    
    # Create combined prompts (question + initial response)
    df['combined_english'] = df[english_prompt_col].astype(str).str.strip() + " " + df[english_initial_col].astype(str).str.strip()
    df['combined_hindi'] = df[hindi_prompt_col].astype(str).str.strip() + " " + df[hindi_initial_col].astype(str).str.strip()
    
    # Remove extra whitespace
    df['combined_english'] = df['combined_english'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['combined_hindi'] = df['combined_hindi'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    logger.info(f"✓ Created combined prompts")
    logger.info(f"Sample combined English (len={len(df['combined_english'].iloc[0])}): {df['combined_english'].iloc[0][:100]}...")
    logger.info(f"Sample combined Hindi (len={len(df['combined_hindi'].iloc[0])}): {df['combined_hindi'].iloc[0][:100]}...")
    
    logger.info("=" * 80)
    logger.info("STEP 3: Setting Up Models")
    logger.info("=" * 80)
    
    # Load base model and tokenizer
    base_model, tokenizer = setup_model_and_tokenizer(
        model_name=base_model_name,
        device_map=device_map,
    )
    
    # Load aligned model (with adapter)
    aligned_model = load_aligned_model(base_model, aligned_model_path)
    
    logger.info("=" * 80)
    logger.info("STEP 4: Generating Predictions for Missing Rows")
    logger.info("=" * 80)
    
    total_rows = len(rows_to_evaluate)
    logger.info(f"Evaluating {total_rows} rows...")
    
    for batch_idx, row_idx in tqdm(enumerate(rows_to_evaluate), total=total_rows, desc="Evaluating"):
        try:
            # Get combined prompts
            english_prompt = df.iloc[row_idx]['combined_english']
            hindi_prompt = df.iloc[row_idx]['combined_hindi']
            
            # Validate prompts
            if not english_prompt or english_prompt.strip() == "":
                logger.warning(f"Row {row_idx}: Empty English prompt, skipping")
                results_df.at[row_idx, "base_english"] = "SKIPPED: Empty prompt"
                continue
            if not hindi_prompt or hindi_prompt.strip() == "":
                logger.warning(f"Row {row_idx}: Empty Hindi prompt, skipping")
                results_df.at[row_idx, "base_hindi"] = "SKIPPED: Empty prompt"
                continue
            
            logger.debug(f"Row {row_idx} - EN prompt (len={len(english_prompt)}): {english_prompt[:80]}...")
            logger.debug(f"Row {row_idx} - HI prompt (len={len(hindi_prompt)}): {hindi_prompt[:80]}...")
            
            # Base model predictions
            logger.debug(f"[{batch_idx+1}/{total_rows}] Generating base model predictions for row {row_idx}")
            base_en = generate_response(base_model, tokenizer, english_prompt)
            base_hi = generate_response(base_model, tokenizer, hindi_prompt)
            
            # Check for empty responses
            if not base_en or base_en.strip() == "":
                logger.warning(f"Row {row_idx}: Base model returned empty English response")
                base_en = "EMPTY_RESPONSE"
            if not base_hi or base_hi.strip() == "":
                logger.warning(f"Row {row_idx}: Base model returned empty Hindi response")
                base_hi = "EMPTY_RESPONSE"
            
            # Aligned model predictions
            logger.debug(f"[{batch_idx+1}/{total_rows}] Generating aligned model predictions for row {row_idx}")
            aligned_en = generate_response(aligned_model, tokenizer, english_prompt)
            aligned_hi = generate_response(aligned_model, tokenizer, hindi_prompt)
            
            # Check for empty responses
            if not aligned_en or aligned_en.strip() == "":
                logger.warning(f"Row {row_idx}: Aligned model returned empty English response")
                aligned_en = "EMPTY_RESPONSE"
            if not aligned_hi or aligned_hi.strip() == "":
                logger.warning(f"Row {row_idx}: Aligned model returned empty Hindi response")
                aligned_hi = "EMPTY_RESPONSE"
            
            # Update the results dataframe
            results_df.at[row_idx, "base_english"] = base_en
            results_df.at[row_idx, "base_hindi"] = base_hi
            results_df.at[row_idx, "aligned_english"] = aligned_en
            results_df.at[row_idx, "aligned_hindi"] = aligned_hi
            
        except Exception as e:
            logger.error(f"Error processing row {row_idx}: {type(e).__name__}: {e}", exc_info=True)
            results_df.at[row_idx, "base_english"] = f"ERROR: {str(e)[:100]}"
            results_df.at[row_idx, "base_hindi"] = f"ERROR: {str(e)[:100]}"
            results_df.at[row_idx, "aligned_english"] = f"ERROR: {str(e)[:100]}"
            results_df.at[row_idx, "aligned_hindi"] = f"ERROR: {str(e)[:100]}"
    
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
