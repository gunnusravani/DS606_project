#!/usr/bin/env python3
"""
Diagnose tokenizer consistency in DPO dataset preparation.

This script identifies why TRL is warning about mismatches between:
- tokenized prompt
- tokenized (prompt + chosen)
- tokenized (prompt + rejected)

Root cause is often decode/re-encode cycles or special token handling.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def analyze_tokenization_mismatch(
    tokenizer,
    prompt: str,
    response: str,
    label: str = "example"
) -> Dict:
    """
    Analyze tokenization for a prompt-response pair to identify mismatches.
    
    This reproduces what DPOTrainer does internally.
    """
    # Method 1: Tokenize prompt separately
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    
    # Method 2: Tokenize prompt in isolation (as DPOTrainer might)
    prompt_with_special = tokenizer(prompt, truncation=False, add_special_tokens=True)
    prompt_ids_method2 = prompt_with_special["input_ids"]
    
    # Method 3: Tokenize full text (prompt + response)
    full_text = prompt + response
    full_tokens = tokenizer.encode(full_text, add_special_tokens=True)
    
    # Method 4: What happens with decode/re-encode cycle (current code issue)
    prompt_decoded_then_retokenized = tokenizer.encode(
        tokenizer.decode(prompt_tokens, skip_special_tokens=True),
        add_special_tokens=True
    )
    
    full_decoded_then_retokenized = tokenizer.encode(
        tokenizer.decode(full_tokens, skip_special_tokens=True),
        add_special_tokens=True
    )
    
    # Check for mismatches
    mismatches = {
        "prompt_tokens_vs_method2": prompt_tokens != prompt_ids_method2,
        "prompt_in_full_text": full_tokens[:len(prompt_tokens)] == prompt_tokens if len(full_tokens) >= len(prompt_tokens) else False,
        "decode_retokenize_mismatch_prompt": prompt_tokens != prompt_decoded_then_retokenized,
        "decode_retokenize_mismatch_full": full_tokens != full_decoded_then_retokenized,
    }
    
    return {
        "label": label,
        "prompt_length": len(prompt),
        "response_length": len(response),
        "prompt_tokens_count": len(prompt_tokens),
        "full_tokens_count": len(full_tokens),
        "prompt_is_prefix_of_full": full_tokens[:len(prompt_tokens)] == prompt_tokens if len(full_tokens) >= len(prompt_tokens) else False,
        "mismatches": mismatches,
        "prompt_tokens_sample": prompt_tokens[:20],
        "full_tokens_prefix_sample": full_tokens[:20],
        "decoded_prompt": tokenizer.decode(prompt_tokens, skip_special_tokens=True)[:100],
        "decoded_full": tokenizer.decode(full_tokens, skip_special_tokens=True)[:100],
    }


def check_tokenizer_settings(tokenizer):
    """Check tokenizer configuration settings."""
    return {
        "pad_token": tokenizer.pad_token,
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "unk_token": tokenizer.unk_token,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "add_special_tokens_default": tokenizer.add_special_tokens,
    }


def extract_prompt_and_response(text: str) -> Tuple[str, str]:
    """
    Extract prompt and response from HH-RLHF format.
    
    Format: "Human: <prompt>\n\nAssistant: <response>"
    """
    if "Assistant:" not in text:
        return text, ""
    
    prompt, response = text.split("Assistant:", 1)
    prompt = prompt.strip()
    response = "Assistant:" + response
    
    return prompt, response


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("\n" + "="*80)
    print("TOKENIZER CONSISTENCY DIAGNOSTIC")
    print("="*80 + "\n")
    
    # Load tokenizer
    model_name = "meta-llama/Llama-3.2-3B"
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    
    # Check tokenizer settings
    print("\n1. TOKENIZER CONFIGURATION:")
    print("-" * 80)
    settings = check_tokenizer_settings(tokenizer)
    for key, value in settings.items():
        print(f"  {key:30} = {value}")
    
    # Load a few examples from HH-RLHF
    print("\n2. LOADING HH-RLHF EXAMPLES:")
    print("-" * 80)
    dataset = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    
    # Analyze first 5 examples
    mismatches_found = 0
    prefixes_mismatch = 0
    
    print("\nAnalyzing first 5 examples...\n")
    
    for idx, example in enumerate(dataset.take(5)):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {idx + 1}: CHOSEN vs REJECTED")
        print(f"{'='*80}")
        
        # Extract prompts and responses
        prompt_c, response_c = extract_prompt_and_response(example["chosen"])
        prompt_r, response_r = extract_prompt_and_response(example["rejected"])
        
        # Analyze chosen
        print(f"\n  Prompt (char length: {len(prompt_c)}):")
        print(f"    {prompt_c[:150]}...")
        
        print(f"\n  Chosen Response (char length: {len(response_c)}):")
        print(f"    {response_c[:150]}...")
        
        analysis_c = analyze_tokenization_mismatch(
            tokenizer, prompt_c, response_c, "chosen"
        )
        
        print(f"\n  [CHOSEN] Tokenization Analysis:")
        print(f"    Prompt tokens: {analysis_c['prompt_tokens_count']}")
        print(f"    Full (prompt+response) tokens: {analysis_c['full_tokens_count']}")
        print(f"    Prompt is prefix of full: {analysis_c['prompt_is_prefix_of_full']}")
        
        if not analysis_c['prompt_is_prefix_of_full']:
            prefixes_mismatch += 1
            print(f"    ⚠️  PREFIX MISMATCH DETECTED!")
            print(f"       Prompt tokens: {analysis_c['prompt_tokens_sample']}")
            print(f"       Full tokens:   {analysis_c['full_tokens_prefix_sample']}")
            print(f"       Decoded prompt: {analysis_c['decoded_prompt']}")
            print(f"       Decoded full:   {analysis_c['decoded_full']}")
        
        # Check for decode/re-encode issues
        if analysis_c['mismatches']['decode_retokenize_mismatch_prompt']:
            mismatches_found += 1
            print(f"    ⚠️  DECODE/RE-ENCODE MISMATCH on prompt!")
        
        if analysis_c['mismatches']['decode_retokenize_mismatch_full']:
            print(f"    ⚠️  DECODE/RE-ENCODE MISMATCH on full text!")
    
    # Summary
    print(f"\n\n{'='*80}")
    print("DIAGNOSTIC SUMMARY:")
    print(f"{'='*80}")
    print(f"  Examples analyzed: 5")
    print(f"  Prefix mismatches found: {prefixes_mismatch}")
    print(f"  Decode/re-encode issues: {mismatches_found}")
    
    if prefixes_mismatch > 0:
        print(f"\n  🔍 ROOT CAUSE ANALYSIS:")
        print(f"     The TRL warnings are triggered because the tokenized prompt")
        print(f"     is NOT a perfect prefix of the tokenized (prompt + response).")
        print(f"\n     This typically happens because:")
        print(f"     1. Special tokens (BOS/EOS) are added/removed differently")
        print(f"     2. Whitespace handling differs between prompt and combined text")
        print(f"     3. The decode/re-encode cycle changes token IDs")
        print(f"\n     CURRENT FIX in hh_rlhf.py:")
        print(f"     - Using decode() then re-encode() on extracted strings")
        print(f"     - This breaks the token sequence alignment")
        print(f"\n     RECOMMENDED FIX:")
        print(f"     - Keep original text, don't decode/re-encode")
        print(f"     - Or use tokenizer.encode_plus() without decode step")
    else:
        print(f"\n  ✅ NO MAJOR MISMATCHES DETECTED")
        print(f"     The tokenizer is processing text consistently!")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
