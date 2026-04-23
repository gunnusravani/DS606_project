# Implementation Summary: Initial Malicious Response Evaluation

## What Was Built

A new evaluation pipeline that combines questions + initial malicious responses and sends them to models for continuation generation.

---

## Feature: Combine Prompt + Initial Response

**Before:** 
- Evaluate on standalone prompts

**Now:**
- Combine `question + intital_malicious_english` → Send to model
- Combine `hindi + intital_malicious_hindi` → Send to model

This simulates a more realistic attack where the adversary starts the model on a harmful path and continues it.

---

## Implementation Details

### 1. New Evaluation Function in `src/ds606/models/evaluate.py`

```python
def evaluate_models_with_initial_response(
    csv_path: str,
    english_prompt_col: str = "question",
    english_initial_col: str = "intital_malicious_english",
    hindi_prompt_col: str = "hindi",
    hindi_initial_col: str = "intital_malicious_hindi",
    ...
) -> None:
    """
    Evaluate models on dataset where prompt is combined with initial malicious response.
    """
```

**Features:**
- Auto-detects and combines columns from your CSV
- Generates responses for both base and aligned models
- Supports resume/continuation of incomplete evaluations
- Saves results with metadata

### 2. New CLI Command

```bash
python -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --device-map cuda:3 \
  --output-dir outputs/evaluations_malicious/
```

**Available flags:**
- `--english-prompt-col` (default: "question")
- `--english-initial-col` (default: "intital_malicious_english")
- `--hindi-prompt-col` (default: "hindi")
- `--hindi-initial-col` (default: "intital_malicious_hindi")
- `--max-samples` (optional: limit for testing)
- `--fill-missing` (default: True, auto-resume)
- `--no-resume` (force re-evaluate all)

### 3. Quick Test Script

```bash
cd /Users/sravani/Documents/VSCode_projects/DS606_project
python scripts/test_evaluate_initial.py
```

This script:
- Inspects your dataset
- Shows sample data
- Asks for confirmation
- Runs evaluation with prompts

---

## How Your Data Flows

```
CSV Input:
┌─────────────────┬──────────────────────┬──────────────────┐
│ question        │ intital_malicious_en │ column name      │
├─────────────────┼──────────────────────┼──────────────────┤
│ "How to hack"   │ "Follow these steps" │ etc...           │
└─────────────────┴──────────────────────┴──────────────────┘
         ↓
COMBINED PROMPT:
  "How to hack Follow these steps"
         ↓
SENT TO MODELS:
  base_model(combined_prompt) → continuation response
  aligned_model(combined_prompt) → continuation response
         ↓
SAVED TO CSV:
┌─────────────┬──────────────────┬────────────────┐
│ question    │ intital_malicious│ base_english   │
├─────────────┼──────────────────┼────────────────┤
│ "How to..." │ "Follow steps"   │ "...response"  │
└─────────────┴──────────────────┴────────────────┘
```

---

## Usage Examples

### Example 1: Quick Test (10 samples)

```bash
python -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --aligned-model outputs/models/dpo/ \
  --device-map cuda:3 \
  --max-samples 10
```

### Example 2: Full Evaluation (all samples)

```bash
python -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --aligned-model outputs/models/dpo/ \
  --device-map cuda:3 \
  --output-dir outputs/evaluations_malicious/
```

### Example 3: Resume Incomplete Evaluation

```bash
# If it crashes, just run the same command again
# It will auto-detect and fill only missing rows
python -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --aligned-model outputs/models/dpo/ \
  --device-map cuda:3
```

### Example 4: Custom Column Names

```bash
# If your CSV has different column names
python -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --aligned-model outputs/models/dpo/ \
  --device-map cuda:3 \
  --english-prompt-col "english_question" \
  --english-initial-col "english_malicious" \
  --hindi-prompt-col "hindi_question" \
  --hindi-initial-col "hindi_malicious"
```

### Example 5: Interactive Python

```python
from src.ds606.models.evaluate import evaluate_models_with_initial_response

evaluate_models_with_initial_response(
    csv_path="initial_malicious_final.csv",
    base_model_name="meta-llama/Meta-Llama-3-8B",
    aligned_model_path="outputs/models/dpo/",
    device_map="cuda:3",
    output_path="outputs/evaluations_malicious/",
    max_samples=100,  # Test with 100 first
)
```

---

## Output Files

### `outputs/evaluations_malicious/malicious_initial_results.csv`

| question | intital_malicious_english | base_english | aligned_english | hindi | intital_malicious_hindi | base_hindi | aligned_hindi |
|----------|--------------------------|--------------|-----------------|-------|------------------------|-----------|---------------|
| "How to..." | "Way to..." | "[response]" | "[response]" | "कैसे..." | "तरीका..." | "[response]" | "[response]" |

**Columns:**
- Original input columns (question, hindi, initial responses)
- 4 new prediction columns:
  - `base_english` - base model's continuation in English
  - `base_hindi` - base model's continuation in Hindi
  - `aligned_english` - aligned model's continuation in English
  - `aligned_hindi` - aligned model's continuation in Hindi

### `outputs/evaluations_malicious/malicious_initial_summary.json`

```json
{
  "total_samples": 1668,
  "rows_evaluated_in_this_run": 1668,
  "base_model": "meta-llama/Meta-Llama-3-8B",
  "aligned_model_path": "outputs/models/dpo/",
  "column_mapping": {
    "english_prompt": "question",
    "english_initial_response": "intital_malicious_english",
    "hindi_prompt": "hindi",
    "hindi_initial_response": "intital_malicious_hindi"
  },
  "resume_enabled": true
}
```

---

## Next Steps

After evaluation completes:

### 1. Translate Hindi Responses (if needed)
```bash
jupyter notebook notebooks/gtrans_attn.ipynb
# Update to use outputs/evaluations_malicious/malicious_initial_results.csv
# Translate: base_hindi → gtrans_base_hindi, aligned_hindi → gtrans_aligned_hindi
```

### 2. Classify Safety
```bash
jupyter notebook notebooks/llamaguard.ipynb
# Use English columns (base_english, aligned_english, gtrans_base_hindi, gtrans_aligned_hindi)
# Classify as SAFE/UNSAFE with categories (S1-S14)
```

### 3. Calculate Metrics
```python
# ASR, CLTG, AGS for prompt+initial vs standalone
import pandas as pd

results = pd.read_csv("outputs/evaluations_malicious/malicious_initial_results.csv")
# Calculate Attack Success Rate per category/language
# Compare with previous evaluation_results.csv to see impact of initial response
```

---

## Files Modified

1. **`src/ds606/models/evaluate.py`** 
   - Added `evaluate_models_with_initial_response()` function
   - 130+ lines of new code

2. **`src/ds606/cli.py`**
   - Added `evaluate-with-initial` subcommand
   - 80+ lines of argument parsing and execution handler

3. **New files created:**
   - `EVALUATE_INITIAL_RESPONSE.md` - Detailed usage guide
   - `scripts/test_evaluate_initial.py` - Interactive test script

---

## Quick Checklist

Before running:

- [ ] Dataset file: `initial_malicious_final.csv` in project root
- [ ] CSV has required columns: question, hindi, intital_malicious_english, intital_malicious_hindi
- [ ] Aligned model exists: `outputs/models/dpo/`
- [ ] GPU available (cuda:3 is specified)
- [ ] Disk space for results (depends on sample count)

After running:

- [ ] Check results: `outputs/evaluations_malicious/malicious_initial_results.csv`
- [ ] Verify no ERROR entries in prediction columns
- [ ] Compare with previous evaluation for differences
- [ ] Proceed to translation/classification as needed

---

## Troubleshooting

### "Column not found" error
- Check your CSV has: `question`, `hindi`, `intital_malicious_english`, `intital_malicious_hindi`
- Or use custom column names with `--english-prompt-col` etc.

### "CUDA out of memory"
- Try different GPU: `--device-map cuda:2`
- Or run on CPU (slower): `--device-map cpu`

### Evaluation very slow
- Generating 1,028 tokens per response takes time
- 10,000+ samples = 8-48 hours depending on GPU
- Start with `--max-samples 10` to test

### Need to restart
- Just run the same command again
- Auto-resume will detect and fill only missing rows

---

## Command Reference

| Task | Command |
|------|---------|
| Quick test (10 samples) | `python -m ds606.cli evaluate-with-initial --csv initial_malicious_final.csv --max-samples 10 --device-map cuda:3` |
| Full evaluation | `python -m ds606.cli evaluate-with-initial --csv initial_malicious_final.csv --device-map cuda:3` |
| Interactive test | `python scripts/test_evaluate_initial.py` |
| Check existing results | `head outputs/evaluations_malicious/malicious_initial_results.csv` |
| Count completed rows | `python -c "import pandas as pd; df=pd.read_csv('outputs/evaluations_malicious/malicious_initial_results.csv'); print(f'{df.dropna(how=\"any\", subset=[\"base_english\", \"base_hindi\", \"aligned_english\", \"aligned_hindi\"]).shape[0]}/{len(df)}' )"` |

---

**For detailed usage:** See `EVALUATE_INITIAL_RESPONSE.md`  
**For interactive testing:** Run `python scripts/test_evaluate_initial.py`
