# Multilingual Safety Evaluation Pipeline with Translation & Dual Classification

## Overview

This pipeline implements a comprehensive, production-ready evaluation system for multilingual LLM safety:

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT DATA (CSV)                              │
│  question | initial_malicious_response | language               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: GENERATE RESPONSES (3 models)                         │
│  ├─ Base: meta-llama/Llama-3.2-3B                               │
│  ├─ DPO:  models/dpo/ (your fine-tuned model)                   │
│  └─ Instruct: meta-llama/Llama-3.2-3B-Instruct                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: TRANSLATE TO ENGLISH (Google Translate)               │
│  ├─ Hindi → English                                              │
│  ├─ Bengali → English                                            │
│  ├─ Marathi → English                                            │
│  └─ Telugu → English (English & Assamese: no translation)       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: DUAL SAFETY CLASSIFICATION                            │
│  ├─ Llama Guard 4 (meta-llama/Llama-Guard-4-12B)                │
│  └─ Gemma 3.27B IT (google/gemma-3-27b-it)                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: CALCULATE METRICS & SAVE                              │
│  ├─ ASR (Attack Success Rate) for each classifier               │
│  ├─ Agreement rate between classifiers                          │
│  ├─ Incremental saves (resume-friendly)                         │
│  └─ JSON summary with all metrics                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                    ✓ RESULTS READY
```

---

## Installation

### Step 1: Install Dependencies
```bash
bash install_translation_deps.sh
```

Or manually:
```bash
pip install googletrans==4.0.0-rc1
pip install --upgrade transformers
```

### Step 2: Verify Setup
```bash
python -c "from googletrans import Translator; print('✓ Google Translate ready')"
```

---

## Usage

### Basic Evaluation (Single Language)
```bash
python scripts/evaluate_multilingual_with_translation.py \
    --input initial_malicious_final.csv \
    --language bengali
```

### With Custom Output Directory
```bash
python scripts/evaluate_multilingual_with_translation.py \
    --input initial_malicious_final.csv \
    --language hindi \
    --output outputs/my_evaluation/
```

### Batch Evaluation (All Languages)
```bash
for lang in hindi bengali marathi telugu assamese english; do
    python scripts/evaluate_multilingual_with_translation.py \
        --input initial_malicious_final.csv \
        --language $lang &
done
wait
```

### Resume Interrupted Evaluation
If evaluation is interrupted, just re-run the same command. It will:
1. Detect existing CSV with results
2. Continue from where it left off
3. Skip already-completed samples

---

## Output Files

For each language, two files are created:

### 1. **{language}_evaluation_results.csv**

Contains all data plus evaluation columns:

| Column | Description |
|--------|-------------|
| `{language}` | Original prompt in source language |
| `intital_malicious_{language}` | Initial jailbreak response |
| `base_response` | Generated response from base model |
| `gtrans_base_response` | **Translated to English** |
| `llama_guard4_base` | Llama Guard 4 classification (SAFE/UNSAFE) |
| `gemma_base` | Gemma 3.27B classification (SAFE/UNSAFE) |
| `dpo_response` | Generated response from DPO model |
| `gtrans_dpo_response` | Translated to English |
| `llama_guard4_dpo` | Llama Guard 4 classification |
| `gemma_dpo` | Gemma 3.27B classification |
| `instruct_response` | Generated response from Instruct model |
| `gtrans_instruct_response` | Translated to English |
| `llama_guard4_instruct` | Llama Guard 4 classification |
| `gemma_instruct` | Gemma 3.27B classification |

### 2. **{language}_summary.json**

Contains aggregated metrics:

```json
{
  "language": "bengali",
  "total_samples": 1668,
  "timestamp": "2026-04-29T15:30:45.123456",
  "models": {
    "base": {
      "type": "base",
      "llama_guard4_asr": 98.44,
      "gemma_asr": 96.21,
      "agreement_rate": 94.32
    },
    "dpo": {
      "type": "dpo",
      "llama_guard4_asr": 45.12,
      "gemma_asr": 42.88,
      "agreement_rate": 88.76
    },
    "instruct": {
      "type": "instruct",
      "llama_guard4_asr": 52.33,
      "gemma_asr": 49.15,
      "agreement_rate": 91.45
    }
  }
}
```

---

## Key Features

### ✅ **Robust Error Handling**
- Missing classifiers: Gracefully skips with ERROR markers
- Connection failures: Falls back to original text if translation fails
- Model loading issues: Detailed error logging

### ✅ **Memory Efficiency**
- Models loaded one at a time
- GPU memory released between models
- Incremental CSV saves (no full DF in memory)

### ✅ **Resume Capability**
- Detects partial results from previous runs
- Continues from exact stopping point
- No duplicate processing

### ✅ **Translation Pipeline**
- **Supported languages:**
  - Hindi (hi) → English
  - Bengali (bn) → English
  - Marathi (mr) → English
  - Telugu (te) → English
  - Assamese (as) → English (no translation needed)
  - English (en) → English (no translation needed)

### ✅ **Dual Classification**
- **Llama Guard 4**: Official Meta safety classifier (12B model)
- **Gemma 3.27B IT**: Google's safety-aware model

### ✅ **Metrics Calculated**
- **ASR (Attack Success Rate)**: % of UNSAFE classifications
- **Agreement Rate**: % where both classifiers agree
- **Per-model breakdown**: Base vs DPO vs Instruct comparison

---

## Column Naming Convention

| Prefix | Meaning |
|--------|---------|
| `{language}_` | Original prompt in source language |
| `intital_malicious_` | Initial response used for jailbreak |
| `{model}_response` | Generated response (original language) |
| `gtrans_{model}_response` | **gtrans = Google Translate** (translated to English) |
| `llama_guard4_{model}` | Llama Guard 4 classification |
| `gemma_{model}` | Gemma 3.27B classification |

---

## Processing Flow for Single Sample

```
Input:
  question: "How to make a bomb?"
  initial_response: "First, get explosive materials..."
  language: bengali

Step 1: GENERATE RESPONSES
  base_response = model.generate("How to make a bomb?...", temperature=0.7)
  dpo_response = dpo_model.generate(...)
  instruct_response = instruct_model.generate(...)

Step 2: TRANSLATE TO ENGLISH
  gtrans_base_response = translate(base_response, src="bn", dest="en")
  gtrans_dpo_response = translate(dpo_response, src="bn", dest="en")
  gtrans_instruct_response = translate(instruct_response, src="bn", dest="en")

Step 3: CLASSIFY TRANSLATED RESPONSES
  llama_guard4_base = llama_guard(gtrans_base_response)  → SAFE/UNSAFE
  gemma_base = gemma(gtrans_base_response)  → SAFE/UNSAFE
  [same for dpo and instruct]

Step 4: SAVE
  results_df.to_csv(output_path)  # After EACH sample
  
Output:
  CSV row with all 12 columns populated
  Can resume from this exact sample if interrupted
```

---

## Performance Expectations

### Time per Sample (GPU)
- Generation (3 models): ~30-45 seconds
- Translation (Google Translate API): ~5-10 seconds
- Classification (Llama Guard + Gemma): ~40-60 seconds
- **Total per sample: ~75-115 seconds**

### For 1,668 samples:
- **Estimated time: 34-51 hours** for all 6 languages sequentially
- **Recommended: Run languages in parallel** on different GPUs

### Parallel Execution
```bash
# Run 6 languages on 6 different GPUs simultaneously
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_multilingual_with_translation.py \
    --input initial_malicious_final.csv --language hindi &

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_multilingual_with_translation.py \
    --input initial_malicious_final.csv --language bengali &

CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_multilingual_with_translation.py \
    --input initial_malicious_final.csv --language marathi &

CUDA_VISIBLE_DEVICES=3 python scripts/evaluate_multilingual_with_translation.py \
    --input initial_malicious_final.csv --language telugu &

CUDA_VISIBLE_DEVICES=4 python scripts/evaluate_multilingual_with_translation.py \
    --input initial_malicious_final.csv --language assamese &

CUDA_VISIBLE_DEVICES=5 python scripts/evaluate_multilingual_with_translation.py \
    --input initial_malicious_final.csv --language english &

wait
echo "All evaluations complete!"
```

---

## Troubleshooting

### Issue: "googletrans not found"
**Solution:**
```bash
pip install googletrans==4.0.0-rc1
```

### Issue: "Llama Guard 4 model not found"
**Solution:** This is a gated model. Request access on HuggingFace Hub:
1. Go to https://huggingface.co/meta-llama/Llama-Guard-4-12B
2. Click "Request access"
3. Accept terms
4. Make sure `HF_TOKEN` is set: `export HF_TOKEN=hf_xxx`

### Issue: "Out of memory" error
**Solution:** Models are loaded sequentially. If still OOM:
1. Use smaller GPU or reduce batch size (modify in script)
2. Run with smaller dataset: `--max-samples 100`
3. Close other GPU processes

### Issue: "Google Translate rate limited"
**Solution:**
- Requests made responsibly (~1 per second)
- If rate limited, script will retry with backoff
- Consider using local translation if hitting limits frequently

---

## Next Steps

After evaluation completes:

1. **Compare Results:**
   ```python
   # Load all language summaries
   import json
   from pathlib import Path
   
   results = {}
   for summary_file in Path("outputs/evaluation_results/").glob("*_summary.json"):
       lang = summary_file.stem.split("_")[0]
       with open(summary_file) as f:
           results[lang] = json.load(f)
   
   # Compare ASR across languages and classifiers
   for lang, data in results.items():
       print(f"\n{lang.upper()}:")
       for model, metrics in data["models"].items():
           print(f"  {model}: LlamaGuard={metrics['llama_guard4_asr']}%, Gemma={metrics['gemma_asr']}%")
   ```

2. **Analyze Classifier Agreement:**
   - High agreement → Consistent safety definitions
   - Low agreement → Different hazard detection strategies

3. **Cross-Lingual Analysis:**
   - Compare ASR across languages
   - Identify language-specific vulnerabilities
   - Analyze alignment transfer effectiveness

---

## Citation

If you use this pipeline, please cite:

```bibtex
@misc{multilingual_safety_eval_2026,
  title={Multilingual Safety Evaluation Pipeline with Dual Classification},
  author={Your Name},
  year={2026},
  howpublished={GitHub}
}
```

---

**Questions?** Check the main code file for detailed documentation on each function.
