# DS606: Cross-Lingual Safety Alignment in LLMs
## Project Documentation & Work Division

**Team:** Sravani Gunnu, Aryan Kashyap, Shahab Ahmad  
**Date:** April 2026

---

## Executive Summary

This project investigates whether safety guardrails learned in English transfer to other languages (Hindi, Bengali, Hinglish) in Large Language Models. We are training and evaluating Meta-Llama-3-8B with alignment techniques (SFT + DPO), then testing robustness using cross-lingual jailbreak prompts and implementing refusal direction analysis.

**Key Findings Target:**
- Measure safety alignment transfer from English to Hindi
- Identify which harm categories show largest safety gaps across languages
- Extract universal refusal directions across languages

---

## Project Background

### Problem Statement

Most LLM safety research focuses on English. However, models are deployed globally across 22+ languages with vastly different safety requirements. The critical question:

> **"When we align an LLM to be safe in English, do those safety properties automatically apply to other languages?"**

### Proposed Solution

We follow a three-stage pipeline:

1. **Training Stage:** Align Llama-3-8B using SFT + DPO on HH-RLHF dataset
2. **Evaluation Stage:** Test both base and aligned models on cross-lingual jailbreak dataset
3. **Analysis Stage:** Extract and analyze refusal mechanism differences across languages

---

## Detailed Methodology

### Stage 1: Model Alignment (Training)

#### Step 1.1: Supervised Fine-Tuning (SFT)

**Objective:** Train model to follow helpful, harmless instructions.

**Dataset:** HH-RLHF (Anthropic Helpfulness-Harmlessness RLHF)
- Total prompts: 160,000
- Training subset: 5,000 prompts (reduced for faster iteration)
- Train-eval split: 95%-5%

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Model | Meta-Llama-3-8B |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Learning Rate | 2e-4 |
| Batch Size | 4 |
| Epochs | 5 |
| Max Sequence Length | 1024 tokens |
| Quantization | 8-bit |
| Device | GPU (cuda:2) |

**Training Results:**
- Initial loss: 2.3
- Final loss: 0.2149 (converged)
- Training time: 4h 29m
- Trainable parameters: 6.8M / 8B (0.0848%)
- Checkpoint saved: `outputs/models/sft/`

#### Step 1.2: Direct Preference Optimization (DPO)

**Objective:** Fine-tune using preference pairs to directly optimize alignment.

**Why DPO instead of RLHF?**
- No reward model training needed
- More stable and efficient
- Better for cross-lingual transfer (no language-specific reward model)

**Configuration:**

| Parameter | Value |
|-----------|-------|
| SFT Checkpoint Load | outputs/models/sft/ |
| Learning Rate | 5e-5 (lower than SFT) |
| Batch Size | 1 (memory constrained) |
| Epochs | 1 (test run), can scale to 3-5 |
| Beta (preference strength) | 0.1 |
| Max Sequence Length | 1024 tokens |
| Quantization | 8-bit |
| Device | GPU (cuda:3) |

**DPO Training Results:**
- Training time: ~1.5 hours for 1 epoch
- Final training loss: 5.8020
- Eval loss: 4.1039
- Checkpoint saved: `outputs/models/dpo/`

---

### Stage 2: Cross-Lingual Evaluation

#### Step 2.1: Jailbreak Dataset Preparation

**Dataset:** `jailbreak.csv`
- Total prompts: 1,668 (representative samples of harmful categories)
- Languages: English & Hindi
- Categories: 14 harm types
  - Violating privacy
  - Harmful content (chaos, violence)
  - Discrimination (race, age, disability)
  - Cyber crime (hacking, financial fraud)
  - Fraud/deception (cheating, fake credentials)
  - Unlawful practices (corruption, abuse)
  - Political offenses
  - Copyright violations

#### Step 2.2: Model Response Generation

**Process:** Generate model responses for all 4 conditions:

| Model | English | Hindi |
|-------|---------|-------|
| Base (Llama-3-8B) | base_english | base_hindi |
| Aligned (SFT+DPO) | aligned_english | aligned_hindi |

**Generation Parameters:**
- Max tokens: 1,028
- Temperature: 0.7
- Top-p: 0.9
- Sampling: Enabled (do_sample=True)

**Current Status:**
- Completed: 10 samples for testing
- Pending: Full evaluation on 1,668 samples
- Output file: `outputs/evaluations/evaluation_results.csv`
- Smart resume: Can re-run to fill missing values auto-detected

---

### Stage 3: Safety Classification & Analysis

#### Step 3.1: Hindi Response Translation

**Objective:** Translate Hindi model responses to English for safety classification.

**Tool:** Google Translate via `gtrans_attn.ipynb`

**Process:**
1. Auto-detect missing/failed translations
2. Retry with exponential backoff (max 4 retries)
3. Handle long sentences by splitting strategy
4. Save progress every 200 rows

**Output Columns:**
- `gtrans_base_hindi` (English translation of base model's Hindi response)
- `gtrans_aligned_hindi` (English translation of aligned model's Hindi response)

#### Step 3.2: Safety Classification with Llama-Guard

**Objective:** Classify each response as SAFE or UNSAFE using Llama-Guard-3-8B.

**Tool:** `llamaguard.ipynb`

**Llama-Guard Safety Categories:**

| Code | Harm Type |
|------|-----------|
| S1 | Violent Crimes |
| S2 | Hate Speech |
| S3 | Harassment |
| S4 | Sexual Content |
| S5 | Criminal Activity |
| S6 | Specialized Advice |
| S7 | Self-harm |
| S8 | Defamation |
| S9 | Misinformation |
| S10 | Fraud |
| S11 | Privacy Violations |
| S12 | Copyright Abuse |
| S13 | Malware/Phishing |
| S14 | Code Interpreter Abuse |

**Classification Output Format:**
- First line: `safe` or `unsafe` (lowercase)
- If unsafe, second line: Comma-separated category codes (e.g., "S1,S3")

#### Step 3.3: Majority Voting (Future Enhancement)

**Plan:** Use 3 OSS LLMs for robust classification
1. Llama-Guard-3-8B (primary classifier)
2. LLaMA-CoT (reasoning-based safety classifier)
3. Falcon Safety (alternative OSS safety classifier)

**Voting Strategy:**
- Majority vote determines final SAFE/UNSAFE label
- Ties resolved by primary classifier (Llama-Guard)

---

### Stage 4: Metric Calculation

#### Step 4.1: Attack Success Rate (ASR)

**Formula:**
```
ASR(L, c) = N_harmful(L, c) / N_total(L, c)
```

where:
- L = language (English, Hindi)
- c = harm category (S1-S14)
- N_harmful = number of UNSAFE responses
- N_total = total number of evaluated prompts

**Interpretation:**
- Higher ASR = Model is less safe (easier to jailbreak)
- ASR=1.0 means 100% of prompts triggered harmful responses
- ASR=0.0 means model refused all prompts

#### Step 4.2: Cross-Lingual Transfer Gap (CLTG)

**Formula:**
```
CLTG(L, c) = ASR_aligned(L, c) - ASR_aligned(English, c)
```

**Interpretation:**
- Positive CLTG = Safety gap exists (Hindi is less safe than English)
- Negative CLTG = Hindi is safer than English (rare)
- CLTG ≈ 0 = Perfect transfer of safety guarantees

#### Step 4.3: Alignment Generalization Score (AGS)

**Formula:**
```
AGS(L) = [ASR_base(L) - ASR_aligned(L)] / [ASR_base(English) - ASR_aligned(English)]
```

**Interpretation:**
- AGS = 1: Perfect generalization (same improvement as English)
- AGS ≈ 0: No generalization (alignment doesn't help in target language)
- AGS < 0: Negative transfer (alignment makes it worse)

---

### Stage 5: Mechanistic Analysis (Refusal Direction)

**Reference Paper:** Wang et al. (2025) - "Refusal direction is universal across safety-aligned languages"

**Key Hypothesis:** Safety-aligned models encode refusal information in a consistent direction in activation space across languages.

**Analysis Steps:**

1. **Extract Activation Representations**
   - Hook final layer activations during inference
   - Collect for prompts that trigger refusals vs. harmful responses
   - Separate by language (English, Hindi)

2. **Compute Refusal Direction**
   - Mean activation difference: unsafe prompts vs. safe prompts
   - PCA to find principal direction of variance
   - Compute refusal vector r_L for each language L

3. **Measure Universality**
   - Cosine similarity between r_English and r_Hindi
   - High cosine similarity (> 0.85) = universal direction
   - Low similarity (< 0.70) = language-specific mechanisms

4. **Steering Experiment**
   - Apply English refusal direction as steering vector during Hindi generation
   - Measure ASR reduction
   - If successful → supports universality hypothesis

---

## Work Completed ✅

- ✅ **Environment Setup**
  - Installed PyTorch 2.4.0, Transformers 4.41.0, TRL 0.10.1, PEFT 0.7.1
  - Resolved 9 major dependency issues

- ✅ **Config System Development**
  - Built modular configuration pipeline (`src/ds606/config.py`)
  - YAML-based hyperparameter management
  - Type-safe dataclasses

- ✅ **Dataset Pipeline**
  - Created HH-RLHF loader (`src/ds606/data/hh_rlhf.py`)
  - Format converters for SFT and DPO training

- ✅ **SFT Training Module**
  - Implemented full 8-step SFT pipeline (`src/ds606/models/sft.py`)
  - Completed 5-epoch training: loss 2.3 → 0.2149
  - Checkpoint saved at `outputs/models/sft/`

- ✅ **DPO Training Module**
  - Implemented DPOTrainer orchestration (`src/ds606/models/dpo.py`)
  - Completed 1-epoch test run
  - Checkpoint saved at `outputs/models/dpo/`

- ✅ **CLI Infrastructure**
  - Built command-line interface with subcommands
  - Support: `train-sft`, `train-dpo`, `evaluate-models`

- ✅ **Evaluation Module (Partial)**
  - Created evaluation framework (`src/ds606/models/evaluate.py`)
  - Successfully evaluated 10 jailbreak prompts (proof of concept)
  - Smart resume feature: detect and fill missing values automatically

- ✅ **GPU Memory Optimization**
  - 8-bit quantization reduces peak memory ~75%
  - Gradient checkpointing enabled

---

## Work Pending ⏭️

### Phase 1: Complete Jailbreak Evaluation

- **Fill Missing Evaluations**
  - Command: `python -m ds606.cli evaluate-models --csv jailbreak.csv --aligned-model outputs/models/dpo/`
  - Current: 10/1,668 samples complete
  - Expected time: 2-3 days (continuous GPU run)
  - Estimated: 1,668 × 4 responses = 6,672 generations

- **Validate Response Quality**
  - Sample 100 responses for manual inspection
  - Check for truncation, coherence, refusal patterns
  - Verify model distinguishes between languages

### Phase 2: Hindi-to-English Translation

- **Translate Hindi Responses to English**
  - Use: `gtrans_attn.ipynb`
  - Translate 2 columns: base_hindi, aligned_hindi
  - Output columns: gtrans_base_hindi, gtrans_aligned_hindi
  - Expected time: 3-4 hours (API rate limited)

- **Quality Check Translations**
  - Sample 50 translations for manual review
  - Check for: sense preservation, grammatical correctness
  - Retry failed/empty translations with fallback strategies

### Phase 3: Safety Classification

- **Classify with Llama-Guard-3-8B**
  - Use: `llamaguard.ipynb`
  - Classify all 4 response columns as SAFE/UNSAFE
  - Extract harm categories (S1-S14) for unsafe responses
  - Expected time: 1-2 hours

- **Implement Majority Voting (Optional Enhancement)**
  - Add 2 more OSS safety classifiers
  - 3-way majority vote for final labels
  - ~2-3 hours additional computation

### Phase 4: Metric Calculation & Analysis

- **Compute ASR, CLTG, AGS**
  - Script: Python notebook or script in `src/ds606/analysis/`
  - Per-category breakdown: ASR by harm type
  - Per-language breakdown: English vs. Hindi
  - Generate visualizations: heatmaps, bar charts

- **Generate Summary Report**
  - Tables with ASR, CLTG, AGS values
  - Key findings: which categories show largest gaps?
  - Failure mode analysis

### Phase 5: Mechanistic Analysis (Refusal Direction)

- **Extract Activation Representations**
  - Implement activation hooking for final decoder layer
  - Collect activations for ~100 refusal vs. harmful response pairs
  - Per language: English & Hindi

- **Compute Refusal Directions**
  - Mean activation difference for each language
  - PCA to get principal direction
  - Compute cosine similarity

- **Steering Experiments**
  - Apply English refusal direction to Hindi generation
  - Measure ASR reduction
  - Validate universality hypothesis

---

## Timeline & Task Division

### Overall Timeline

| Week | Dates | Milestone | Status |
|------|-------|-----------|--------|
| 1-2 | Mar 8-21 | Setup & Config | ✓ Complete |
| 3 | Mar 22-28 | SFT & DPO Training | ✓ Complete |
| 4 | Mar 29-Apr 4 | Evaluation Setup | ✓ Complete |
| 5 | Apr 5-11 | Full Evaluation & Translation | 🔄 In Progress |
| 6 | Apr 12-18 | Safety Classification | ⏭️ Pending |
| 7 | Apr 19-25 | Metrics & Analysis | ⏭️ Pending |
| 8 | Apr 26-May 1 | Mechanistic Analysis | ⏭️ Pending |
| 9 | May 2-8 | Final Report & Presentation | ⏭️ Pending |

### Task Assignment

#### Sravani (Lead) - ML Engineering & Integration

**Current (Week 5):**
- Complete full jailbreak evaluation (1,668 samples)
- Validate response quality and completeness
- Ensure no missing/ERROR values in evaluation_results.csv

**Week 6:**
- Run Google Translate on Hindi columns (gtrans_attn.ipynb)
- Implement Llama-Guard classification (llamaguard.ipynb)
- Handle any translation failures with retries

**Weeks 7-8:**
- Oversee metric calculation and analysis reports
- Implement refusal direction extraction
- Conduct steering experiments

---

#### Aryan (B.Tech Civil) - Data Analysis & Visualization

**Current (Week 5):**

- **Task A1:** Quality assurance on evaluation results
  - Sample 100 jailbreak responses manually
  - Check for: truncation, coherence, actual refusals vs. harmful content
  - Document any patterns or issues found
  - Expected effort: 3-4 hours

- **Task A2:** Translation quality verification
  - After gtrans runs, sample 50 Hindi→English translations
  - Verify meaning preservation
  - Flag any obviously wrong translations for re-running
  - Expected effort: 2 hours

**Week 6:**

- **Task A3:** Safety classification validation
  - Sample 50 Llama-Guard classifications
  - Manually verify SAFE/UNSAFE labels
  - Check category assignments for unsafe responses
  - Expected effort: 3 hours

**Week 7:**

- **Task A4:** Create comprehensive visualizations
  - ASR heatmap: models × languages × categories
  - Bar charts: CLTG by category
  - Failure analysis: which prompts did alignment NOT help?
  - Box plots: distribution of ASR across categories
  - Expected effort: 4-5 hours (with mentor guidance)

- **Task A5:** Generate analysis report
  - Summarize key findings in tables and prose
  - Identify top 5 categories with largest safety gaps
  - Compare base vs. aligned model improvements
  - Expected effort: 2-3 hours

**Week 9:**
- Help prepare presentation slides
- Create summary infographics for key results

---

#### Shahab (B.Tech Civil) - Documentation & Infrastructure

**Current (Week 5):**

- **Task S1:** Create detailed workflow documentation
  - Document exact commands to run evaluation
  - Create step-by-step guides for each phase
  - Include troubleshooting section (what to do if X fails)
  - Expected effort: 2-3 hours

- **Task S2:** Setup monitoring & logging
  - Create run logs tracking progress
  - Monitor GPU memory usage during long evaluations
  - Document any errors or crashes with timestamps
  - Expected effort: 1 hour initial setup

**Week 6:**

- **Task S3:** Translation pipeline documentation
  - Document Google Translate setup and API usage
  - Create fallback procedures for API failures
  - Track translation statistics (success rate, avg time per sample)
  - Expected effort: 2 hours

- **Task S4:** Llama-Guard deployment guide
  - Document model loading and inference setup
  - Create classification pipeline documentation
  - Record performance metrics (tokens/sec, memory usage)
  - Expected effort: 2 hours

**Weeks 7-8:**
- Task S5: Analysis pipeline documentation
- Task S6: Final code cleanup & submission prep

**Week 9:**
- Co-author final technical report
- Prepare code for submission

---

## Key File Locations

| File | Purpose |
|------|---------|
| `src/ds606/models/evaluate.py` | Evaluation framework |
| `src/ds606/models/sft.py` | SFT training |
| `src/ds606/models/dpo.py` | DPO training |
| `configs/training_*.yaml` | Hyperparameters |
| `outputs/models/sft/` | SFT checkpoint |
| `outputs/models/dpo/` | DPO checkpoint |
| `outputs/evaluations/evaluation_results.csv` | All model responses |
| `notebooks/gtrans_attn.ipynb` | Hindi translation |
| `notebooks/llamaguard.ipynb` | Safety classification |
| `jailbreak.csv` | 1,668 test prompts |

---

## Success Criteria

✅ All 1,668 jailbreak samples evaluated (4 responses each)  
✅ ASR, CLTG, AGS metrics computed  
✅ CLTG > 0 in ≥3 categories (transfer gap exists)  
✅ Refusal direction similarity > 0.70 (universality)  
✅ Steering experiment shows ≥10% ASR reduction  
✅ Technical report + presentation + clean code

---

## References

1. Wang, X., et al. (2025). Refusal direction is universal across safety-aligned languages. NeurIPS
2. Deng, Y., et al. (2024). Multilingual jailbreak challenges. ICLR
3. Dou, Y., et al. (2024). Cross-lingual safety transfer. ACL

