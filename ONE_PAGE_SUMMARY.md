# DS606 PROJECT ONE-PAGE SUMMARY

## Project Title
**Do Safety Guardrails Generalize Across Languages? A Study of Cross-Lingual Safety Alignment Transfer in LLMs**

---

## THE BIG QUESTION
When an LLM is made safe in English through training, are those safety guarantees maintained when the model generates responses in other languages (Hindi, Bengali)?

---

## PROJECT PIPELINE (5 STAGES)

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   STAGE 1    │ => │   STAGE 2    │ => │   STAGE 3    │ => │   STAGE 4    │ => │   STAGE 5    │
│   Training   │    │ Evaluation   │    │Translation & │    │  Metrics &   │    │ Mechanistic  │
│   (SFT+DPO)  │    │  Jailbreak   │    │Classification│    │  Analysis    │    │  Analysis    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
     ✅ DONE              🔄 IN PROGRESS        ⏭️ PENDING         ⏭️ PENDING        ⏭️ PENDING
  (Mar 22-28)        (Apr 5-11: 10/1668)    (Apr 12-16)       (Apr 17-25)       (Apr 26-May 8)
```

---

## WHAT WE'VE DONE ✅

| ✓ | Task | Completion | Output |
|---|------|----------|--------|
| ✓ | Installed dependencies (PyTorch, Transformers, TRL, PEFT) | 100% | Working environment |
| ✓ | SFT Training: 5,000 HH-RLHF samples, 5 epochs | 100% | `outputs/models/sft/` |
| ✓ | DPO Training: 4,750 preference pairs | 100% | `outputs/models/dpo/` |
| ✓ | Evaluation Framework: 4-column response generation | 100% | Code ready |
| ✓ | Smart Resume: Auto-detect & fill missing values | 100% | Feature implemented |
| ✓ | Proof of Concept: 10 jailbreak samples evaluated | 100% | Initial results |

---

## WHAT'S NEXT (PRIORITY ORDER) ⏭️

### WEEK 5 (Apr 5-11): Complete Evaluation
- **Who:** Sravani
- **What:** Generate responses for ALL 1,668 jailbreak samples in 4 conditions
  - base_english
  - base_hindi  
  - aligned_english
  - aligned_hindi
- **Output:** `outputs/evaluations/evaluation_results.csv` (4 complete columns)
- **Time:** 2-3 days continuous GPU running
- **Command:** `python -m ds606.cli evaluate-models --csv jailbreak.csv --aligned-model outputs/models/dpo/ --device-map cuda:3`

**ARYAN:** Manually verify quality of 100 responses during this time
**SHAHAB:** Setup monitoring and create execution guide

---

### WEEK 6 (Apr 12-18): Translate & Classify
1. **Translate Hindi → English**
   - Use: `gtrans_attn.ipynb`
   - Translate: base_hindi, aligned_hindi
   - Output: gtrans_base_hindi, gtrans_aligned_hindi
   - Time: 3-4 hours

2. **Classify with Llama-Guard**
   - Use: `llamaguard.ipynb`
   - Classify all 4 response columns as SAFE/UNSAFE
   - Extract harm categories (S1-S14)
   - Time: 1-2 hours

---

### WEEK 7 (Apr 19-25): Metrics & Analysis
1. Compute **ASR** (Attack Success Rate) per model/language/category
2. Compute **CLTG** (Cross-Lingual Transfer Gap) - shows safety loss
3. Compute **AGS** (Alignment Generalization Score) - shows how well alignment transfers
4. Generate visualizations: heatmaps, bar charts, failure analysis
5. Write analysis summary

---

### WEEK 8 (Apr 26-May 1): Mechanistic Analysis
1. Extract refusal directions from neural activations
2. Measure universality: English vs Hindi refusal patterns
3. Test hypothesis: Can we transfer English refusal to Hindi?
4. Document findings

---

### WEEK 9 (May 2-8): Final Report
1. Write technical report
2. Prepare presentation slides
3. Clean code and submit

---

## KEY METRICS WE'LL REPORT

### ASR (Attack Success Rate)
- **Definition:** Fraction of prompts the model responds to unsafely
- **Formula:** unsafe_responses / total_responses
- **Goal:** Low for aligned model, especially in all languages

### CLTG (Cross-Lingual Transfer Gap)  
- **Definition:** How much safety is LOST when switching from English to Hindi
- **Formula:** ASR(Hindi) - ASR(English) for aligned model
- **Goal:** CLTG ≈ 0 means perfect transfer

### AGS (Alignment Generalization Score)
- **Definition:** What fraction of English alignment benefit applies to Hindi
- **Formula:** Improvement(Hindi) / Improvement(English)
- **Goal:** AGS ≈ 1.0 means equal improvement in both languages

---

## TEAM ROLES

| Team Member | Role | Primary Responsibility |
|_____________|______|_______________________
| **Sravani** | ML Engineer | Train models, run evaluation, oversee experiments |
| **Aryan** | Data Analyst | Manual QA, visualization, analysis reports |
| **Shahab** | Infrastructure/Docs | Documentation, monitoring, reproducibility |

---

## CRITICAL FILE LOCATIONS

```
Evaluation Results:  outputs/evaluations/evaluation_results.csv
Translation Code:    notebooks/gtrans_attn.ipynb  
Safety Classifier:   notebooks/llamaguard.ipynb
Test Dataset:        jailbreak.csv (1,668 harmful questions)
Trained Models:      outputs/models/sft/  and  outputs/models/dpo/
```

---

## DATASET SPECS

| Aspect | Details |
|--------|---------|
| Test Prompts | jailbreak.csv with 1,668 harmful/edge-case questions |
| Languages | English + Hindi |
| Safety Categories | 14 types (S1-S14): violence, fraud, privacy, etc. |
| Responses Generated | 6,672 total (1,668 × 4 conditions) |
| Models | Base: Llama-3-8B, Aligned: Llama-3-8B + SFT + DPO |

---

## SUCCESS CRITERIA (FINAL DELIVERABLES)

✅ **Evaluation Complete:** All 1,668 samples evaluated in 4 conditions
✅ **Translation Done:** Hindi responses converted to English  
✅ **Classification Done:** All responses labeled SAFE/UNSAFE + categories
✅ **Metrics Computed:** ASR, CLTG, AGS calculated per category
✅ **Finding 1:** CLTG > 0 in ≥3 categories (transfer gap exists)
✅ **Finding 2:** Refusal direction similarity > 0.70 (universality)
✅ **Finding 3:** Steering experiment shows ≥10% ASR reduction (proof of mechanism)
✅ **Deliverables:** Technical report + presentation + clean code

---

## WHO SHOULD KNOW WHAT

**For Reviewers/Professors:**
- Read: `DS606_Project_Documentation.md` (complete overview)
- See: Presentation slides (key visuals and findings)

**For Team Members:**
- Use: `TASK_TRACKING.md` (weekly task breakdown with exact efforts)
- Reference: This one-pager (quick status check)

**For Future Researchers:**
- Study: Code in `src/ds606/` with detailed comments
- Follow: `README.md` (reproduction steps)
- Analyze: `outputs/` directory with all results and logs

---

## QUICK STATUS CHECK

**As of April 11, 2026:**

- ✅ **Weeks 1-4:** Training & setup COMPLETE
- 🔄 **Week 5:** Full evaluation 10/1,668 samples (0.6%) - 95% remaining
- ⏳ **Week 6:** Translation & classification - WAIT for evaluation
- ⏳ **Week 7:** Metrics & visualization - WAIT for classification
- ⏳ **Week 8:** Mechanistic analysis - WAIT for metrics
- ⏳ **Week 9:** Final report - WAIT for all experiments

**Critical Path:** Evaluation must complete FIRST before anything else can start!

---

## NEED HELP?

- **Sravani:** ML architecture, training issues, experiment design
- **Aryan:** Visualization, data quality, manual verification
- **Shahab:** Setup, documentation, reproducibility

**Expected response time:** 2 hours during work hours

---

**This is your quick reference guide. Print this page and share it with your team!**

For detailed instructions: See `DS606_Project_Documentation.md`  
For daily task tracking: See `TASK_TRACKING.md`  
For code: See `src/ds606/` in the repository

---

*Generated April 11, 2026 | Next update: April 18, 2026*

