# DS606 Project - Task Tracking & Quick Reference

## 📋 QUICK STATUS (April 11, 2026)

| Item | Status | Timeline |
|------|--------|----------|
| **Training** | ✅ COMPLETE | Mar 22-28 |
| **10-Sample Eval** | ✅ COMPLETE | Apr 1-4 |
| **Full Evaluation** | 🔄 IN PROGRESS | Apr 5-11 |
| **Translation** | ⏭️ PENDING | Apr 12-16 |
| **Safety Classification** | ⏭️ PENDING | Apr 17-21 |
| **Metrics & Analysis** | ⏭️ PENDING | Apr 22-30 |
| **Mechanistic Analysis** | ⏭️ PENDING | May 1-8 |
| **Final Report** | ⏭️ PENDING | May 9-14 |

---

## 🎯 IMMEDIATE PRIORITIES (Week 5: Apr 5-11)

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

## 📅 DETAILED WEEKLY BREAKDOWN

### WEEK 5 (Apr 5-11) - CURRENT WEEK
**Goal:** Complete full evaluation + validate quality

#### Sravani's Tasks
1. Evaluate all 1,668 samples (4 variations each = 6,672 responses)
2. Auto-fill missing values using smart resume mode
3. Verify CSV integrity (no NULLs or errors)

#### Aryan's Tasks
1. QA sample 100 base_english responses
2. QA sample 100 aligned_english responses
3. Create QA summary report

#### Shahab's Tasks
1. Setup monitoring infrastructure
2. Create execution guide with exact commands
3. Document any issues encountered

**Success Criteria:** evaluation_results.csv contains all 1,668 rows with 4 complete columns

---

### WEEK 6 (Apr 12-18)
**Goal:** Translate Hindi → English + classify safety

#### Sravani's Tasks
1. Run gtrans_attn.ipynb to translate Hindi columns
   - Input: base_hindi, aligned_hindi
   - Output: gtrans_base_hindi, gtrans_aligned_hindi
   - Handle retries and failures
2. Run llamaguard.ipynb to classify all responses
   - 4 passes (base_en, aligned_en, gtrans_base_hi, gtrans_aligned_hi)
   - Output: 8 safety columns

#### Aryan's Tasks
1. QA sample 50 translated Hindi responses
   - Check meaning preservation
   - Verify no truncation
2. QA sample 50 Llama-Guard classifications
   - Manually verify SAFE/UNSAFE labels
   - Check category assignments

#### Shahab's Tasks
1. Document translation pipeline
   - Google Translate setup
   - API rate limiting strategies
   - Failure handling procedures
2. Document safety classification pipeline
   - Model loading process
   - Category mapping reference

**Success Criteria:** 
- evaluation_results.csv has 4 → 6 → 8 columns progressively
- All safety classifications completed

---

### WEEK 7 (Apr 19-25)
**Goal:** Compute metrics and generate analysis

#### Sravani's Tasks
1. Create metrics calculation module (`src/ds606/analysis/metrics.py`)
   - Implement ASR formula
   - Implement CLTG formula
   - Implement AGS formula
2. Generate per-category breakdowns
3. Identify top 5 categories with largest safety gaps

#### Aryan's Tasks
1. Create comprehensive visualizations
   - **Visualization 1:** ASR heatmap (models × languages × categories)
   - **Visualization 2:** CLTG bar chart by category
   - **Visualization 3:** AGS comparison
   - **Visualization 4:** Failure analysis (which prompts resisted alignment?)
   - **Visualization 5:** Distribution plots
2. Write analysis report with figure captions

#### Shahab's Tasks
1. Document analysis pipeline
2. Create reproducible analysis scripts
3. Generate table summaries for report

**Success Criteria:**
- ASR, CLTG, AGS computed and validated
- 5+ publication-quality visualizations created
- Key findings documented

---

### WEEK 8 (Apr 26-May 1)
**Goal:** Mechanistic analysis - refusal directions

#### Sravani's Tasks
1. Implement activation extraction hook
2. Compute refusal directions (English + Hindi)
3. Measure cosine similarity
4. Run steering experiments
5. Document findings

#### Aryan's Tasks
1. Create visualization of refusal directions
2. Plot activation space projections
3. Create comparison plots between languages

#### Shahab's Tasks
1. Document mechanistic analysis methodology
2. Create reproducible steering experiment script
3. Prepare methodology section for paper

**Success Criteria:**
- Refusal direction cosine similarity computed
- Steering experiment results documented
- Hypothesis validated or refuted with evidence

---

### WEEK 9 (May 2-8)
**Goal:** Final report & presentation

#### All Team Members
1. Sravani: Lead writing + final experiments
2. Aryan: Create presentation slides + infographics
3. Shahab: Final code cleanup + README
4. **Team Sync:** Finalize findings and presentation narrative

**Outputs:**
- Technical report (5-6 pages)
- Presentation slides (15-20 slides)
- Cleaned codebase on GitHub
- Reproducibility checklist

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

