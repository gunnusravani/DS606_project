# Ready to Evaluate with Initial Malicious Responses ✓

## What You Can Do Now

Your dataset `initial_malicious_final.csv` has:
- English questions + Hindi translations
- Initial malicious responses (English & Hindi)

The new evaluation system **combines** them and sends to models:

```
English Prompt:  question + intital_malicious_english
Hindi Prompt:    hindi + intital_malicious_hindi
```

---

## 3 Ways to Run

### 🚀 QUICKEST: Interactive Test Script

```bash
cd /Users/sravani/Documents/VSCode_projects/DS606_project
python scripts/test_evaluate_initial.py
```

This will:
1. Check your dataset
2. Show sample first row
3. Ask confirmation
4. Run evaluation
5. Save results automatically

---

### 🎯 RECOMMENDED: Command Line

```bash
# Test with 10 samples first
python -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --device-map cuda:3 \
  --max-samples 10

# Then run full evaluation
python -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --device-map cuda:3 \
  --output-dir outputs/evaluations_malicious/
```

---

### 📓 INTERACTIVE: Jupyter / IPython

```python
from src.ds606.models.evaluate import evaluate_models_with_initial_response

evaluate_models_with_initial_response(
    csv_path="initial_malicious_final.csv",
    aligned_model_path="outputs/models/dpo/",
    device_map="cuda:3",
    output_path="outputs/evaluations_malicious/",
)
```

---

## Exact Column Names Expected

| Column | Your File | Used For |
|--------|-----------|----------|
| English Question | `question` | Base question |
| Hindi Question | `hindi` | Hindi base question |
| English Initial | `intital_malicious_english` | Initial harmful response (English) |
| Hindi Initial | `intital_malicious_hindi` | Initial harmful response (Hindi) |

✓ All exact names match your CSV

---

## What Gets Generated

After running, you'll have:

```
outputs/evaluations_malicious/
├── malicious_initial_results.csv         ← Main output (predictions)
└── malicious_initial_summary.json        ← Metadata
```

**Results CSV structure:**
```
Original columns: question, hindi, intital_malicious_english, intital_malicious_hindi
+ NEW columns:
  - base_english          [model continuation]
  - base_hindi            [model continuation]
  - aligned_english       [model continuation]
  - aligned_hindi         [model continuation]
```

---

## Step-by-Step Guide

### Step 1: Quick Test (Recommended First)

```bash
python -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --device-map cuda:3 \
  --max-samples 10
```

**Expected:**
- Takes ~2-5 minutes
- Outputs "Evaluating 10 rows... [████████] 10/10"
- Creates `outputs/evaluations_malicious/malicious_initial_results.csv` with 10 rows

**Verify:**
```bash
head outputs/evaluations_malicious/malicious_initial_results.csv
```

Should show your question columns + 4 new prediction columns filled

---

### Step 2: Full Evaluation (Overnight)

```bash
python -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --device-map cuda:3
```

**Expected:**
- Takes 8-48 hours depending on sample count
- Safe to interrupt (use Ctrl+C)
- Auto-resumes from where it stopped

**Monitor progress:**
```bash
# In another terminal
while true; do
  python -c "
    import pandas as pd
    df = pd.read_csv('outputs/evaluations_malicious/malicious_initial_results.csv')
    completed = df[['base_english', 'base_hindi', 'aligned_english', 'aligned_hindi']].dropna(how='any').shape[0]
    print(f'Progress: {completed}/{len(df)} ({100*completed/len(df):.1f}%)')
  "
  sleep 60
done
```

---

### Step 3: Auto-Resume if Interrupted

Just run the same command again:

```bash
python -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --device-map cuda:3
```

It will:
- Detect already-completed rows
- Fill only missing rows
- Skip already-evaluated samples

---

## After Evaluation: Next Steps

### Option A: Translate Hindi Responses
```bash
jupyter notebook notebooks/gtrans_attn.ipynb
# Use: outputs/evaluations_malicious/malicious_initial_results.csv
# Translate: base_hindi → gtrans_base_hindi, aligned_hindi → gtrans_aligned_hindi
```

### Option B: Classify Safety
```bash
jupyter notebook notebooks/llamaguard.ipynb
# Use English columns to classify as SAFE/UNSAFE
# Into categories S1-S14 (violence, fraud, privacy, etc.)
```

### Option C: Compare with Previous Results
```python
import pandas as pd

# Load both evaluations
prev = pd.read_csv("outputs/evaluations/evaluation_results.csv")
curr = pd.read_csv("outputs/evaluations_malicious/malicious_initial_results.csv")

# Compare (roughly, after aligning rows)
print("Base model response length:")
print(f"  Without initial: {prev['base_english'].str.len().mean():.0f} chars avg")
print(f"  With initial:    {curr['base_english'].str.len().mean():.0f} chars avg")

print("\nAligned model response length:")
print(f"  Without initial: {prev['aligned_english'].str.len().mean():.0f} chars avg")
print(f"  With initial:    {curr['aligned_english'].str.len().mean():.0f} chars avg")
```

---

## Pre-Run Checklist

Before starting, verify:

```bash
# 1. Dataset exists
ls -lh initial_malicious_final.csv

# 2. Has required columns
python -c "
import pandas as pd
df = pd.read_csv('initial_malicious_final.csv')
cols = ['question', 'hindi', 'intital_malicious_english', 'intital_malicious_hindi']
print('Columns:', df.columns.tolist())
print('Required columns present:', all(c in df.columns for c in cols))
print('Sample count:', len(df))
"

# 3. Models available
ls -la outputs/models/dpo/

# 4. GPU available
nvidia-smi
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: initial_malicious_final.csv` | Place CSV in project root directory |
| `Column 'question' not found` | Check column spelling: exactly `question` (lowercase) |
| `CUDA out of memory` | Try `--device-map cuda:2` or `cuda:1` |
| `Model not found` | SFT/DPO checkpoints not trained yet |
| Evaluation very slow | Normal - 1,028 tokens per response takes time |
| Wants to re-evaluate everything | Use `--no-resume` flag |

---

## Configuration Options

```bash
python -m ds606.cli evaluate-with-initial \
  --csv <dataset.csv>                              # Required
  --aligned-model <path_to_dpo_or_sft>            # Default: outputs/models/dpo/
  --device-map <cuda:0|cuda:1|...>                # Default: auto
  --output-dir <output_directory>                 # Default: outputs/evaluations_malicious/
  --max-samples <number>                          # Optional: test with limited samples
  --english-prompt-col <col_name>                 # Default: question
  --english-initial-col <col_name>                # Default: intital_malicious_english
  --hindi-prompt-col <col_name>                   # Default: hindi
  --hindi-initial-col <col_name>                  # Default: intital_malicious_hindi
  --fill-missing                                  # Default: True (auto-resume)
  --no-resume                                     # Force re-evaluate all
```

---

## What This Achieves

✓ **Realistic Attack Scenario**
- Not just asking harmful question
- Starting model on harmful path, then asking to continue
- Tests model's resilience to context

✓ **Bilingual Comparison**  
- See if model is more/less robust in Hindi
- Identify language-specific vulnerabilities

✓ **Base vs Aligned**
- Measure DPO training effectiveness
- On this harder attack scenario

✓ **Data for Analysis**
- Can calculate ASR (Attack Success Rate)
- Compare aligned vs base
- Compare English vs Hindi

---

## Expected Results Format

After completion, view first 5 rows:

```bash
head -6 outputs/evaluations_malicious/malicious_initial_results.csv
```

You should see:
- Original columns (question, hindi, initial responses)
- 4 new columns with model-generated continuations
- No ERROR or empty values (unless max-samples was too high)

---

## Questions / Need Help?

- **Detailed guide:** See `EVALUATE_INITIAL_RESPONSE.md`
- **Implementation details:** See `INITIAL_RESPONSE_IMPLEMENTATION.md`
- **Code location:** `src/ds606/models/evaluate.py` function `evaluate_models_with_initial_response()`

---

## TL;DR - Just Run This

```bash
cd /Users/sravani/Documents/VSCode_projects/DS606_project

# Option 1: Interactive test (easiest)
python scripts/test_evaluate_initial.py

# Option 2: Direct command (10 samples to test)
python -m ds606.cli evaluate-with-initial --csv initial_malicious_final.csv --max-samples 10 --device-map cuda:3

# Option 3: Full evaluation (leave overnight)
python -m ds606.cli evaluate-with-initial --csv initial_malicious_final.csv --device-map cuda:3
```

Pick one, run it, and you're done! 🚀

