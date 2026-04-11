# DS606: Cross-Lingual Safety Alignment in LLMs

**Research Question:** Do safety guardrails learned in English transfer to other languages?

This project trains and evaluates Llama-3-8B with alignment techniques (SFT + DPO), testing safety robustness across English and Hindi using cross-lingual jailbreak prompts.

## Quick Start

**For team members:**
- 📖 Read: [DS606_Project_Documentation.md](DS606_Project_Documentation.md) for full overview
- ✅ Track: [TASK_TRACKING.md](TASK_TRACKING.md) for current tasks
- 📄 Reference: [ONE_PAGE_SUMMARY.md](ONE_PAGE_SUMMARY.md) for quick lookup

**For running evaluation:**
```bash
python -m ds606.cli evaluate-models --csv jailbreak.csv --aligned-model outputs/models/dpo/
```

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for detailed instructions.

## Project Structure

```
src/ds606/
├── models/
│   ├── sft.py           # SFT training
│   ├── dpo.py           # DPO training
│   └── evaluate.py      # Evaluation framework
├── data/
│   └── hh_rlhf.py       # Dataset loading
└── cli.py               # Command-line interface

outputs/
├── models/
│   ├── sft/             # SFT checkpoint
│   └── dpo/             # DPO checkpoint
└── evaluations/
    └── evaluation_results.csv  # Model responses

notebooks/
├── gtrans_attn.ipynb    # Hindi → English translation
└── llamaguard.ipynb     # Safety classification
```

## 5-Stage Pipeline

1. **Train** (✅ Done) - SFT + DPO on 5K HH-RLHF samples
2. **Evaluate** (🔄 Progress) - Generate responses for 1,668 jailbreak prompts
3. **Translate & Classify** (⏭️ Next) - Hindi→English + LlamaGuard
4. **Metrics** (⏭️ After) - Compute ASR, CLTG, AGS
5. **Mechanistic** (⏭️ Final) - Extract refusal directions

## Key Metrics

- **ASR:** Attack Success Rate (fraction of unsafe responses)
- **CLTG:** Cross-Lingual Transfer Gap (safety loss in target language)
- **AGS:** Alignment Generalization Score (benefit transfer rate)

## Team

- **Sravani Gunnu** - ML Engineering
- **Aryan Kashyap** - Data Analysis & Visualization  
- **Shahab Ahmad** - Infrastructure & Documentation

## Timeline

| Week | Milestone | Status |
|------|-----------|--------|
| 5 (Apr 5-11) | Full Evaluation | 🔄 In Progress |
| 6 (Apr 12-18) | Translation & Classification | ⏭️ Pending |
| 7 (Apr 19-25) | Metrics & Analysis | ⏭️ Pending |
| 8 (Apr 26-May 1) | Mechanistic Analysis | ⏭️ Pending |
| 9 (May 2-8) | Final Report | ⏭️ Pending |

## Dependencies

```
torch==2.4.0
transformers==4.41.0
peft==0.7.1
trl==0.10.1
bitsandbytes>=0.41.0
googletrans==4.0.0-rc1
```

Install with: `pip install -r requirements.txt`

## References

Wang, X., et al. (2025). Refusal direction is universal across safety-aligned languages. NeurIPS