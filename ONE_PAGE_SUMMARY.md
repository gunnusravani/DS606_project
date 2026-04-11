# DS606 PROJECT - ONE-PAGE SUMMARY

**Full documentation:** [DS606_Project_Documentation.md](DS606_Project_Documentation.md)  
**Task tracking:** [TASK_TRACKING.md](TASK_TRACKING.md)

---

## The Question
When an LLM is made safe in English, do those safety guarantees hold in other languages (Hindi)?

---

## 5-Stage Pipeline

| Stage | Status | Output |
|-------|--------|--------|
| **1. Train** (SFT+DPO) | ✅ Done | `outputs/models/dpo/` |
| **2. Evaluate** (1,668 prompts × 4 conditions) | 🔄 In Progress | evaluation_results.csv |
| **3. Translate & Classify** (Hindi→English + LlamaGuard) | ⏭️ Next | +2 translation cols + 4 safety cols |
| **4. Compute Metrics** (ASR, CLTG, AGS) | ⏭️ After #3 | Analysis report + visualizations |
| **5. Mechanistic** (Refusal directions) | ⏭️ Final | Universality findings |

---

## Team Roles

| Person | Role | Current Task |
|--------|------|--------------|
| **Sravani** | ML Lead | Complete full evaluation (1,668 samples) |
| **Aryan** | Data Analysis | QA 100 responses + prepare visualizations |
| **Shahab** | Infrastructure | Setup monitoring + document pipelines |

---

## Critical Commands

**Start full evaluation (2-3 days):**
```bash
python -m ds606.cli evaluate-models --csv jailbreak.csv --aligned-model outputs/models/dpo/ --device-map cuda:3
```

**Monitor GPU:**
```bash
watch -n 5 nvidia-smi
```

---

## Success Metrics

✅ All 1,668 samples evaluated  
✅ ASR, CLTG, AGS calculated (shows safety gap transfer)  
✅ Refusal direction similarity > 0.70 (universality)  
✅ Steering experiment successful (>10% ASR reduction)

---

## Key Files

- `jailbreak.csv` - Test dataset (1,668 prompts)
- `outputs/evaluations/evaluation_results.csv` - All model responses
- `notebooks/gtrans_attn.ipynb` - Hindi translation
- `notebooks/llamaguard.ipynb` - Safety classification
- `src/ds606/models/evaluate.py` - Evaluation code

---

**Timeline:** 5 weeks (Apr 5-May 8) | **Report Due:** May 8

