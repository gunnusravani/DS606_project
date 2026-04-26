#!/usr/bin/env python3
"""
Language-specific evaluation script for multilingual safety alignment.

Usage:
    python scripts/evaluate_per_language.py --input initial_malicious_final.csv --language hindi
    python scripts/evaluate_per_language.py --input initial_malicious_final.csv --language bengali
    # etc...

This script:
1. Takes a CSV with multilingual columns
2. Filters to specific language
3. Evaluates on 3 models: base, DPO-aligned, and instruct
4. Saves results incrementally (after each model)
5. Resumes from existing results if they exist
"""

import argparse
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
from tqdm import tqdm

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# Load environment
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# LANGUAGE MAPPING
# ============================================================================
LANGUAGE_COLUMNS = {
    "hindi": ("hindi", "intital_malicious_hindi"),
    "bengali": ("bengali", "intital_malicious_bengali"),
    "marathi": ("marathi", "intital_malicious_marathi"),
    "telugu": ("telegu", "intital_malicious_telugu"),
    "assamese": ("assamese", "intital_malicious_english"),  # Use English initial for Assamese
    "english": ("english", "intital_malicious_english"),
}

# Model configurations
MODELS = {
    "base": "meta-llama/Llama-3.2-3B",
    "dpo": "models/dpo/",
    "instruct": "meta-llama/Llama-3.2-3B-Instruct",
}


def load_model_and_tokenizer(model_path: str, device_map: str = "auto"):
    """Load model and tokenizer with Llama 3.2 compatibility."""
    logger.info(f"Loading model: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with rope_scaling fix for Llama 3.2
    use_sdpa = "3.2" not in model_path
    
    try:
        if use_sdpa:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
        else:
            logger.info("Loading Llama 3.2 without SDPA...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
    except ValueError as e:
        if "rope_scaling" in str(e):
            logger.warning("SDPA failed, retrying without it...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            raise
    
    # Load LoRA adapter if DPO model
    if "dpo" in model_path and "models/dpo" in model_path:
        logger.info("Loading DPO LoRA adapter...")
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
    
    model.eval()
    logger.info(f"✓ Model loaded: {model.config.model_type}")
    
    return model, tokenizer


def generate_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> str:
    """Generate response using instruction format."""
    try:
        if not prompt or prompt.strip() == "":
            return "ERROR: Empty prompt"
        
        model.eval()
        
        # Wrap with instruction format
        formatted_prompt = f"""### Instruction:
Read the following text and respond appropriately.

### Input:
{prompt}

### Response:
"""
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        input_ids_len = inputs['input_ids'].shape[1]
        
        # Truncate if too long
        max_prompt_tokens = 5000
        if input_ids_len > max_prompt_tokens:
            inputs['input_ids'] = inputs['input_ids'][:, :max_prompt_tokens]
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'][:, :max_prompt_tokens]
            input_ids_len = max_prompt_tokens
        
        # Generate
        context_limit = 8000
        safe_max_new_tokens = min(512, context_limit - input_ids_len - 50)
        
        if safe_max_new_tokens < 50:
            return f"ERROR: Prompt too long"
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=safe_max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Extract response
        response_tokens = outputs[0][input_ids_len:]
        if len(response_tokens) == 0:
            return "ERROR: No tokens generated"
        
        response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        
        # Extract after "### Response:" marker
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        if not response:
            return "ERROR: Empty response after extraction"
        
        return response
        
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)[:60]}"


def evaluate_language(
    csv_path: str,
    language: str,
    device_map: str = "auto",
):
    """Evaluate all models for a specific language."""
    
    logger.info("=" * 80)
    logger.info(f"LANGUAGE EVALUATION: {language.upper()}")
    logger.info("=" * 80)
    
    # Validate language
    if language not in LANGUAGE_COLUMNS:
        raise ValueError(f"Language '{language}' not supported. Choose from: {list(LANGUAGE_COLUMNS.keys())}")
    
    lang_col, initial_col = LANGUAGE_COLUMNS[language]
    
    # Load CSV
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Filter columns
    logger.info(f"Extracting columns for language: {language}")
    results_df = df[["num", "question", "category", "sub_category"]].copy()
    results_df["language"] = language
    results_df[f"{language}_text"] = df[lang_col].fillna("")
    results_df[f"{language}_initial"] = df[initial_col].fillna("").str[:512]  # Trim to 512 chars
    
    # Create combined prompt column
    results_df["prompt"] = results_df["question"].str.strip() + " " + results_df[f"{language}_initial"].str.strip()
    results_df["prompt"] = results_df["prompt"].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    logger.info(f"Sample prompt: {results_df['prompt'].iloc[0][:100]}...")
    
    # Setup output directory
    output_dir = Path("outputs/llama3.2_3b")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{language}_initial_results.csv"
    
    # Check for existing results and resume
    if output_file.exists():
        logger.info(f"Found existing results: {output_file}")
        results_df = pd.read_csv(output_file)
        logger.info(f"Resuming from {len(results_df)} rows")
    else:
        # Initialize result columns
        for model_name in MODELS.keys():
            results_df[f"{model_name}_response"] = ""
    
    # Evaluate each model
    for model_name, model_path in MODELS.items():
        response_col = f"{model_name}_response"
        
        # Check if already evaluated
        if response_col in results_df.columns and results_df[response_col].notna().sum() > 0:
            completed = results_df[response_col].notna().sum()
            logger.info(f"✓ Model '{model_name}' already evaluated ({completed}/{len(results_df)} rows)")
            continue
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"EVALUATING MODEL: {model_name.upper()}")
        logger.info(f"Model: {model_path}")
        logger.info("=" * 80)
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(model_path, device_map=device_map)
        
        # Generate responses
        responses = []
        for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc=f"Generating {model_name}"):
            if pd.notna(row[response_col]) and row[response_col] != "":
                # Already has response
                responses.append(row[response_col])
            else:
                # Generate new response
                prompt = row["prompt"]
                response = generate_response(model, tokenizer, prompt)
                responses.append(response)
        
        # Update results
        results_df[response_col] = responses
        
        # Save after each model
        results_df.to_csv(output_file, index=False)
        logger.info(f"✓ Saved results to {output_file}")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ EVALUATION COMPLETE")
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 80)
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Language-specific multilingual safety evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/evaluate_per_language.py --input initial_malicious_final.csv --language hindi
    python scripts/evaluate_per_language.py --input initial_malicious_final.csv --language bengali
    python scripts/evaluate_per_language.py --input initial_malicious_final.csv --language marathi
        """,
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="initial_malicious_final.csv",
        help="Input CSV file path (default: initial_malicious_final.csv)",
    )
    
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help=f"Language to evaluate. Options: {', '.join(LANGUAGE_COLUMNS.keys())}",
    )
    
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping for models (default: auto)",
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    try:
        results_df = evaluate_language(
            csv_path=args.input,
            language=args.language,
            device_map=args.device_map,
        )
        logger.info("✓ Successfully completed evaluation!")
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
