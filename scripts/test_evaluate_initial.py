#!/usr/bin/env python3
"""
Quick test script for evaluating models with initial malicious responses.
Run this to get started: python scripts/test_evaluate_initial.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    print("=" * 80)
    print("DS606: Evaluate Models with Initial Malicious Responses - Quick Test")
    print("=" * 80)
    
    # Check if dataset exists
    dataset_path = "initial_malicious_final.csv"
    if not Path(dataset_path).exists():
        print(f"❌ Error: {dataset_path} not found!")
        print(f"   Please ensure the CSV is in the current directory")
        return
    
    print(f"✓ Found dataset: {dataset_path}\n")
    
    # Load and inspect dataset
    print("Step 1: Inspecting dataset...")
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path)
        print(f"  Total samples: {len(df)}")
        print(f"  Columns: {df.columns.tolist()}\n")
        
        # Check required columns
        required_cols = ["question", "hindi", "intital_malicious_english", "intital_malicious_hindi"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            print(f"   Expected columns: {required_cols}")
            return
        
        print(f"  ✓ All required columns present\n")
        
        # Show sample
        print("Sample data (first row):")
        print(f"  Question: {df['question'].iloc[0][:80]}...")
        print(f"  Initial English: {df['intital_malicious_english'].iloc[0][:80]}...")
        print(f"  Hindi: {df['hindi'].iloc[0][:80]}...")
        print(f"  Initial Hindi: {df['intital_malicious_hindi'].iloc[0][:80]}...\n")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Check models
    print("Step 2: Checking model availability...")
    
    # Check if DPO model exists
    dpo_path = "outputs/models/dpo/"
    if not Path(dpo_path).exists():
        print(f"⚠️  Warning: DPO model not found at {dpo_path}")
        print(f"   Will attempt to download base model only")
        use_aligned = False
    else:
        print(f"  ✓ DPO model found at {dpo_path}")
        use_aligned = True
    
    print(f"  ✓ Will use base model: meta-llama/Meta-Llama-3-8B (will download if needed)\n")
    
    # Get user confirmation
    print("Step 3: Configuration")
    print(f"  Dataset: {dataset_path}")
    print(f"  Use aligned model: {use_aligned}")
    print(f"  Output directory: outputs/evaluations_malicious/")
    print(f"  Device: cuda:3")
    
    response = input("\nProceed with evaluation? (y/n): ").strip().lower()
    
    if response != "y":
        print("Evaluation cancelled.")
        return
    
    # Run evaluation
    print("\nStep 4: Starting evaluation...")
    print("=" * 80)
    
    try:
        from ds606.models.evaluate import evaluate_models_with_initial_response
        
        # Ask for sample size
        sample_size = input("How many samples to evaluate? (press Enter for all): ").strip()
        max_samples = int(sample_size) if sample_size else None
        
        evaluate_models_with_initial_response(
            csv_path=dataset_path,
            base_model_name="meta-llama/Meta-Llama-3-8B",
            aligned_model_path="outputs/models/dpo/" if use_aligned else None,
            device_map="cuda:3",
            output_path="outputs/evaluations_malicious/",
            max_samples=max_samples,
            english_prompt_col="question",
            english_initial_col="intital_malicious_english",
            hindi_prompt_col="hindi",
            hindi_initial_col="intital_malicious_hindi",
        )
        
        print("\n" + "=" * 80)
        print("✓ Evaluation completed successfully!")
        print("=" * 80)
        print("\nResults saved to:")
        print("  - outputs/evaluations_malicious/malicious_initial_results.csv")
        print("  - outputs/evaluations_malicious/malicious_initial_summary.json")
        
    except ImportError as e:
        print(f"❌ Error: Could not import evaluation module")
        print(f"   Make sure you're running from the project root directory")
        print(f"   Error: {e}")
        return
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
