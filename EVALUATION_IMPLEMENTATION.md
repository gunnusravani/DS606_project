# Evaluation Module Implementation Summary

## What Was Created

### 1. **evaluate.py** (`src/ds606/models/evaluate.py`)
A comprehensive evaluation module with the following components:

#### Key Functions:
- `setup_model_and_tokenizer()`: Loads base model with 8-bit quantization
- `load_aligned_model()`: Loads LoRA adapter onto base model
- `generate_response()`: Generates model responses with configurable temperature/top_p
- `evaluate_models()`: Main orchestrator that:
  - Loads CSV data
  - Sets up both base and aligned models
  - Generates predictions for both English and Hindi prompts
  - Saves results with model_language naming convention
  - Outputs evaluation_results.csv and evaluation_summary.json

#### Features:
- **Memory Efficient**: 8-bit quantization reduces peak memory usage
- **Error Handling**: Gracefully handles generation errors with detailed logging
- **Flexible**: Supports custom model paths, max samples, and output directories
- **Comprehensive**: Generates 4 prediction columns per sample (2 models × 2 languages)

### 2. **CLI Integration** (Updated `src/ds606/cli.py`)
Added new `evaluate-models` subcommand with arguments:
- `--csv`: Path to evaluation CSV file
- `--base-model`: Base model name (default: meta-llama/Meta-Llama-3-8B)
- `--aligned-model`: LoRA adapter path (default: outputs/models/dpo/)
- `--device-map`: Device mapping (default: auto)
- `--output-dir`: Output directory (default: outputs/evaluations/)
- `--max-samples`: Optional limit on samples to evaluate

### 3. **Documentation** (`EVALUATION_GUIDE.md`)
Complete user guide including:
- CSV format requirements
- Command examples (full evaluation, quick test, custom devices)
- Output file descriptions
- Performance expectations
- Sample analysis code in Python

## Usage

### Basic Evaluation (All Samples):
```bash
python -m ds606.cli evaluate-models \
  --csv jailbreak_just_question.csv \
  --base-model meta-llama/Meta-Llama-3-8B \
  --aligned-model outputs/models/dpo/
```

### Quick Test (First 10 Samples):
```bash
python -m ds606.cli evaluate-models \
  --csv jailbreak_just_question.csv \
  --aligned-model outputs/models/dpo/ \
  --max-samples 10
```

## Output Structure

### evaluation_results.csv
Original CSV columns plus:
- `base_english`: Base model's response to English prompt
- `base_hindi`: Base model's response to Hindi prompt
- `aligned_english`: Aligned model's response to English prompt
- `aligned_hindi`: Aligned model's response to Hindi prompt

Example row:
```
num | English | hindi | category | base_english | base_hindi | aligned_english | aligned_hindi
1   | How would I... | मैं भारत में... | violating_privacy | [response] | [response] | [response] | [response]
```

### evaluation_summary.json
Metadata:
```json
{
  "total_samples_evaluated": 20,
  "base_model": "meta-llama/Meta-Llama-3-8B",
  "aligned_model_path": "outputs/models/dpo/",
  "output_columns": ["base_english", "base_hindi", "aligned_english", "aligned_hindi"],
  "csv_path": "jailbreak_just_question.csv"
}
```

## Performance Characteristics

| Aspect | Details |
|--------|---------|
| **Time per sample** | ~1-2 minutes (4 generations) |
| **Samples per hour** | 30-60 |
| **Memory usage** | ~20-25 GiB peak (both models loaded) |
| **Quantization** | 8-bit for efficiency |
| **Device** | Configurable (GPU:0, GPU:3, auto, etc.) |

## Next Steps

1. **Run a quick test** with `--max-samples 10` to verify setup
2. **Review outputs** in `outputs/evaluations/evaluation_results.csv`
3. **Run full evaluation** on all samples when ready
4. **Analyze results** using comparison metrics (length, refusal rate, etc.)

## Integration with Training Pipeline

```
Meta-Llama-3-8B (Base Model)
         ↓
   [SFT Training]
         ↓
  SFT Checkpoint (outputs/models/sft/)
         ↓
   [DPO Training]
         ↓
  DPO Checkpoint (outputs/models/dpo/)
         ↓
   [EVALUATION] ← You are here
         ↓
  Both models compared on English & Hindi
```
