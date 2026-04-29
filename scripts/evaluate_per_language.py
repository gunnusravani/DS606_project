#!/usr/bin/env python3
"""
Language-specific evaluation script for multilingual safety alignment.

Usage:
    python scripts/evaluate_per_language.py --language hindi
    python scripts/evaluate_per_language.py --language bengali --use-llama-guard
"""

import argparse
import logging
import os
import sys
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
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# TYPES AND MODELS
# ============================================================================
LANGUAGE_COLUMNS = {
    "hindi": ("hindi", "intital_malicious_hindi"),
    "bengali": ("bengali", "intital_malicious_bengali"),
    "marathi": ("marathi", "intital_malicious_marathi"),
    "telugu": ("telegu", "intital_malicious_telugu"),
    "assamese": ("assamese", "intital_malicious_english"),
    "english": ("question", "intital_malicious_english"),
}

MODELS = {
    "base": "meta-llama/Llama-3.2-3B",
    "dpo": "models/dpo/",
    "instruct": "meta-llama/Llama-3.2-3B-Instruct",
}


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model_and_tokenizer(model_path: str, device_map: str = "auto"):
    """Load model and tokenizer with Llama 3.2 compatibility."""
    logger.info(f"Loading: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model (skip SDPA for Llama 3.2)
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
    if "dpo" in model_path.lower() and Path(model_path).exists():
        logger.info("Loading DPO LoRA adapter...")
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
    
    model.eval()
    logger.info(f"✓ Model loaded")
    return model, tokenizer


# ============================================================================
# RESPONSE GENERATION
# ============================================================================
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
            return "ERROR: Prompt too long"
        
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
            return "ERROR: Empty response"
        
        return response
        
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)[:60]}"


# ============================================================================
# LLAMA GUARD CLASSIFICATION
# ============================================================================
def classify_with_llama_guard(responses: list, batch_size: int = 4) -> list:
    """Classify responses using Llama Guard 3."""
    logger.info(f"\nClassifying {len(responses)} responses with Llama Guard...")
    
    try:
        from src.ds606.eval.llama_guard import LlamaGuardClassifier
        
        classifier = LlamaGuardClassifier(model_name="meta-llama/Llama-Guard-3-8B")
        results = classifier.classify_batch(responses, batch_size=batch_size)
        
        # Log stats
        unsafe_count = sum(1 for r in results if r.get("label") == "unsafe")
        asr = (unsafe_count / len(results)) * 100 if results else 0
        logger.info(f"✓ Llama Guard ASR: {asr:.2f}% ({unsafe_count}/{len(results)} unsafe)")
        
        return results
    
    except Exception as e:
        logger.error(f"Llama Guard failed: {e}")
        logger.warning("Skipping Llama Guard classification")
        return None


# ============================================================================
# MAIN EVALUATION
# ============================================================================
def evaluate_language(csv_path: str, language: str, use_llama_guard: bool = False):
    """Evaluate all models for a specific language."""
    
    logger.info("=" * 80)
    logger.info(f"LANGUAGE EVALUATION: {language.upper()}")
    logger.info("=" * 80)
    
    # Validate language
    if language not in LANGUAGE_COLUMNS:
        raise ValueError(f"Language '{language}' not supported. Choose: {list(LANGUAGE_COLUMNS.keys())}")
    
    lang_col, initial_col = LANGUAGE_COLUMNS[language]
    
    # Load CSV
    logger.info(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Prepare data
    results_df = df[["num", "question", "category", "sub_category"]].copy()
    results_df["language"] = language
    results_df[f"{language}_text"] = df[lang_col].fillna("")
    results_df[f"{language}_initial"] = df[initial_col].fillna("").str[:512]
    results_df["prompt"] = (
        results_df["question"].str.strip() + " " + 
        results_df[f"{language}_initial"].str.strip()
    ).str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Setup output
    output_dir = Path("outputs/llama3.2_3b")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{language}_initial_results.csv"
    
    # Check for existing results
    if output_file.exists():
        logger.info(f"Resuming from: {output_file}")
        results_df = pd.read_csv(output_file)
    else:
        # Initialize
        for model_name in MODELS.keys():
            results_df[f"{model_name}_response"] = ""
            if use_llama_guard:
                results_df[f"{model_name}_safety_label"] = ""
                results_df[f"{model_name}_safety_score"] = 0.0
        results_df.to_csv(output_file, index=False)
        logger.info(f"Created: {output_file}")
    
    # Evaluate each model
    for model_name, model_path in MODELS.items():
        response_col = f"{model_name}_response"
        
<<<<<<< Updated upstream
        # Check if already done
=======
        # Check if already evaluated
        print("Not nan values:",results_df[response_col].notna().sum())
        print(results_df[response_col])
>>>>>>> Stashed changes
        if response_col in results_df.columns and results_df[response_col].notna().sum() > 0:
            logger.info(f"✓ {model_name}: already evaluated")
            if use_llama_guard:
                safety_col = f"{model_name}_safety_label"
                if safety_col in results_df.columns and results_df[safety_col].notna().sum() > 0:
                    logger.info(f"✓ {model_name}: Llama Guard already done")
                    continue
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"MODEL: {model_name.upper()}")
        logger.info(f"Path: {model_path}")
        logger.info("=" * 80)
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # Generate responses
        logger.info("Generating responses...")
        responses = []
        for idx, row in tqdm(results_df.iterrows(), total=len(results_df)):
            if pd.notna(row.get(response_col)) and row.get(response_col) != "":
                responses.append(row[response_col])
            else:
                prompt = row["prompt"]
                response = generate_response(model, tokenizer, prompt)
                responses.append(response)
        
        results_df[response_col] = responses
        
        # Apply Llama Guard if enabled
        if use_llama_guard:
            safety_label_col = f"{model_name}_safety_label"
            safety_score_col = f"{model_name}_safety_score"
            
            # Ensure columns exist
            if safety_label_col not in results_df.columns:
                results_df[safety_label_col] = ""
                results_df[safety_score_col] = 0.0
            
            # Classify
            classifications = classify_with_llama_guard(responses, batch_size=4)
            
            if classifications:
                results_df[safety_label_col] = [c.get("label", "unknown") for c in classifications]
                results_df[safety_score_col] = [c.get("score", 0.0) for c in classifications]
        
        # Save
        results_df.to_csv(output_file, index=False)
        logger.info(f"✓ Saved: {output_file}")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ EVALUATION COMPLETE")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Language-specific safety evaluation")
    
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=list(LANGUAGE_COLUMNS.keys()),
        help="Language to evaluate",
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="initial_malicious_final.csv",
        help="Input CSV file",
    )
    
    parser.add_argument(
        "--use-llama-guard",
        action="store_true",
        help="Enable Llama Guard safety classification",
    )
    
    args = parser.parse_args()
    
    try:
        evaluate_language(
            csv_path=args.input,
            language=args.language,
            use_llama_guard=args.use_llama_guard,
        )
        logger.info("✓ Evaluation successful!")
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
