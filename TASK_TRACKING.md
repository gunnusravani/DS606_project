# DS606 Project - Task Tracking

**Status (Apr 11):** Training ✅ | Eval ✅/🔄 | Translation ⏭️ | Classification ⏭️ | Analysis ⏭️

For methodology & background: See [DS606_Project_Documentation.md](DS606_Project_Documentation.md)

---

## 🎯 CURRENT WEEK (Week 5: Apr 5-11)

### Sravani's Priority List
- [ ] Complete evaluation loop on **all 1,668 samples** (currently only 10 done)
- [ ] Verify no missing values in evaluation_results.csv
- [ ] Prepare Hindi translation inputs for gtrans_attn.ipynb

**Command to run:**
```bash
cd /workspace/DS606_project
python -m ds606.cli evaluate-models \
  --csv jailbreak.csv \
  --aligned-model outputs/models/dpo/ \
  --device-map cuda:3
```

**Expected:** 2-3 days runtime (GPU continuous)  
**Output:** `outputs/evaluations/evaluation_results.csv` with 6,672 responses

---

### Aryan's Priority List
- [ ] **Task A1:** Manual QA on first 100 responses
  - Check if responses are coherent
  - Verify English vs Hindi differences
  - Note any truncation or errors
  - Reference file: `outputs/evaluations/evaluation_results.csv`
  - Time: 3-4 hours
  - Output: QA_report_A1.txt

- [ ] Create sample inspection checklist
  - What makes a good response?
  - What are common failure patterns?
  - Example questions to ask while reviewing

**Template for QA Report:**
```
Response ID: [number]
Prompt Language: [EN/HI]
Base Model Response Length: [tokens]
Aligned Model Response Length: [tokens]
Quality Issues Found:
- Issue 1: [description]
- Issue 2: [description]
Notes: [any observations]
```

---

### Shahab's Priority List
- [ ] **Task S1:** Create execution & monitoring guide
  - Document exact commands for each phase
  - Create troubleshooting FAQ
  - Setup monitoring for GPU usage
  - Time: 2-3 hours
  - Output: EXECUTION_GUIDE.md

- [ ] **Task S2:** Setup progress tracking sheet
  - Create shared log for evaluation progress
  - Track errors and issues
  - Document timing information

**Monitoring Commands:**
```bash
# Monitor GPU usage during evaluation
watch -n 5 nvidia-smi

# Check disk space
df -h /workspace/DS606_project/outputs/

# Monitor CPU/Memory
top
```

---

## 📅 UPCOMING WEEKS

### Week 6 (Apr 12-18): Translation & Safety
- **Sravani:** Run gtrans_attn.ipynb + llamaguard.ipynb
- **Aryan:** QA translations + classification labels
- **Shahab:** Document pipelines

### Week 7 (Apr 19-25): Metrics & Analysis
- **Sravani:** Compute ASR, CLTG, AGS metrics
- **Aryan:** Create visualizations + write report
- **Shahab:** Finalize documentation

### Week 8 (Apr 26-May 1): Mechanistic Analysis
- **Sravani:** Extract refusal directions + steering experiments
- **All:** Support as needed

### Week 9 (May 2-8): Final Report
- **All:** Prepare presentation + clean code + submit

---

## 📅 DETAILED WEEKLY BREAKDOWN

### WEEK 5 (Apr 5-11) - CURRENT WEEK
**Goal:** Complete full evaluation + validate quality

#### Sravani's Tasks
1. Evaluate all 1,668 samples (4 variations each = 6,672 responses)

---

## 📊 DATASET SIZES & MEMORY REQUIREMENTS

| Stage | Input Rows | Computations | Est. Time | GPU Memory |
|-------|-----------|--------------|-----------|-----------|
| Full Evaluation | 1,668 | 6,672 responses | 2-3 days | 8-12GB |
| Translation | 3,336 | API calls | 3-4 hours | CPU (minimal) |
| Safety Classification | 3,336 | 4 inference passes | 1-2 hours | 8-12GB |
| Analysis | 1,668 | Metrics computation | 1 hour | CPU (minimal) |
| Refusal Direction | 1,668 | Activation extraction | 2-3 hours | 8-12GB |

---

## 🔧 CRITICAL COMMANDS REFERENCE

### Start Full Evaluation
```bash
cd /workspace/DS606_project
python -m ds606.cli evaluate-models \
  --csv jailbreak.csv \
  --aligned-model outputs/models/dpo/ \
  --device-map cuda:3 \
  --output-dir outputs/evaluations/
```

### Run Translation (Interactive Jupyter)
```bash
jupyter notebook notebooks/gtrans_attn.ipynb
# Update: root_directory = "./outputs/evaluations/"
# Update: column_pairs = [("base_hindi", "gtrans_base_hindi"), ...]
# Run all cells
```

### Run Safety Classification (Interactive Jupyter)
```bash
export HF_TOKEN="hf_your_token_here"
jupyter notebook notebooks/llamaguard.ipynb
# Update: DATA_PATH = "./outputs/evaluations/evaluation_results.csv"
# Run all cells
```

### Monitor GPU While Running
```bash
# Terminal 1: Run evaluation
cd /workspace/DS606_project
python -m ds606.cli evaluate-models ...

# Terminal 2: Monitor GPU (refresh every 5 seconds)
watch -n 5 nvidia-smi
```

### Check Progress (Smart Resume)
```bash
# This will show how many rows still need evaluation
python -c "
import pandas as pd
df = pd.read_csv('outputs/evaluations/evaluation_results.csv')
print(f'Total rows: {len(df)}')
print(f'Complete rows (all 4 cols): {df.dropna(subset=[\"base_english\", \"base_hindi\", \"aligned_english\", \"aligned_hindi\"]).shape[0]}')
print(f'Missing rows: {df[[\"base_english\", \"base_hindi\", \"aligned_english\", \"aligned_hindi\"]].isna().any(axis=1).sum()}')
"
```

---

## ⚠️ COMMON PITFALLS & SOLUTIONS

| Problem | Cause | Solution |
|---------|-------|----------|
| Evaluation crashes after 100 samples | GPU memory accumulation | Restart kernel/process every 500 samples |
| Translation fails on long responses | API timeout on large text | Split long responses in gtrans code |
| Llama-Guard loads slowly | Model quantization delay | Cache model after first load |
| Missing values in CSV | Process interrupted | Use smart resume mode (automatic) |
| API rate limit exceeded | Too many requests | Add `time.sleep(2)` between batches |

---

## 📝 TEAM COMMUNICATION PROTOCOL

### Daily Standup (15 minutes, ~10 AM)
**Each person updates:**
1. What did I complete yesterday?
2. What am I working on today?
3. Any blockers or issues?

### Progress Log (Update weekly)
**File:** `PROGRESS_LOG.md` (to be created in repo)
```markdown
## Week 5 Progress
- [x] Task X by Sravani (completed Apr 8)
- [x] Task Y by Aryan (completed Apr 9)
- [ ] Task Z by Shahab (blocked on X)
```

### Issue Reporting
**If you encounter an error:**
1. Document the error message
2. Note what command triggered it
3. Report in team chat with filepath and timestamp
4. Sravani will help debug

---

## 💾 FILE LOCATIONS & OUTPUTS

| Component | Location | Owner |
|-----------|----------|-------|
| Training code | `src/ds606/models/` | Sravani |
| Evaluation code | `src/ds606/models/evaluate.py` | Sravani |
| Config files | `configs/` | Sravani |
| Evaluation results | `outputs/evaluations/evaluation_results.csv` | Sravani |
| Translation notebook | `notebooks/gtrans_attn.ipynb` | Sravani/Shahab |
| Classification notebook | `notebooks/llamaguard.ipynb` | Sravani/Shahab |
| Analysis code | `src/ds606/analysis/` | Aryan/Sravani |
| Visualizations | `outputs/figures/` | Aryan |
| QA Reports | `outputs/qa_reports/` | Aryan |
| Documentation | `outputs/documentation/` | Shahab |

---

## 🎓 LEARNING RESOURCES FOR TEAM

**For Aryan (Data Analysis):**
- matplotlib tutorials: https://matplotlib.org/tutorials/
- seaborn heatmaps: https://seaborn.pydata.org/generated/seaborn.heatmap.html
- pandas groupby: https://pandas.pydata.org/docs/user_guide/groupby.html

**For Shahab (Documentation):**
- Markdown guide: https://www.markdownguide.org/
- GitHub README best practices: https://www.makeareadme.com/
- Bash scripting: https://www.shellscript.sh/

**For Everyone:**
- LLM safety: https://arxiv.org/abs/2310.10136
- Cross-lingual NLP: https://aclanthology.org/
- Mechanistic interpretability: http://neuron2vec.org/

---

## ✅ FINAL DELIVERABLES CHECKLIST

Before submission, verify:

- [ ] All 1,668 jailbreak prompts have responses in 4 conditions
- [ ] Hindi responses translated to English
- [ ] All responses classified as SAFE/UNSAFE with categories
- [ ] ASR computed per model, language, and category
- [ ] CLTG computed showing safety gaps
- [ ] AGS computed showing generalization
- [ ] Refusal direction cosine similarity computed (> 0.70 for universality)
- [ ] Steering experiment results documented
- [ ] 5+ publication-quality figures generated
- [ ] Technical report written (5-6 pages)
- [ ] Presentation slides ready (15-20 slides)
- [ ] Code cleaned, commented, and in GitHub
- [ ] README with reproduction instructions
- [ ] All results saved in `outputs/` directory

---

## 📞 CONTACT & ESCALATION

**If stuck on:**
- **Setup/Installation issues** → Sravani
- **Data quality questions** → Aryan  
- **Documentation/Logging questions** → Shahab
- **Conceptual questions** → Sravani (lead)

**Expected response time:** Within 2 hours during work hours

---

**Last Updated:** April 11, 2026  
**Next Review:** April 18, 2026 (end of Week 6)

