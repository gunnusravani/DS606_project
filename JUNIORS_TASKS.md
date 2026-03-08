# Initial tasks for junior contributors

## Junior A — Data + prompt variants + validation (Week 1)
**Goal:** Provide a validated, unified prompt dataset (JSONL) incl. Hinglish variants.

### Tasks
1. **Define prompt schema**
   - Implement required fields in `src/ds606/data/schema.py`:
     - `id` (str), `language` (en/hi/bn/hinglish), `variant` (original/hi_clean/hinglish_natural/hinglish_phonetic),
       `harm_category` (enum), `prompt` (str), `source` (xsafety/indicsafe/india_harms), optional `notes`.
   - Add `validate_prompt(record) -> None` that raises `ValueError` with helpful messages.

2. **JSONL IO**
   - Implement `read_jsonl(path) -> list[dict]` and `write_jsonl(path, rows) -> None` in `src/ds606/io/jsonl.py`.

3. **Dataset loader + merge**
   - Implement `load_prompts(paths: list[str]) -> list[dict]` in `src/ds606/data/load.py`
     - reads JSONL, validates, de-duplicates by `id`, returns list.

4. **Variant generation utility (lightweight)**
   - Add `src/ds606/data/augment.py` (optional) or extend `load.py` with helpers that:
     - given a base record, create 3 variants by copying and changing `language/variant/prompt`.
   - For now, start with a small hand-made mapping for ~10 prompts (no LLM-generation).

5. **Unit tests**
   - Add tests under `tests/`:
     - schema validation catches missing fields
     - loader de-dupes correctly

### Deliverables
- `data/prompts/india_harms.jsonl` with at least 20 example prompts (balanced across categories)
- Passing tests

---

## Junior B — Evaluation metrics + refusal detection + stats (Week 1)
**Goal:** Given generations, compute ASR/Refusal/CLTG/AGS with CIs.

### Tasks
1. **Refusal detector**
   - Implement `is_refusal(text: str) -> bool` in `src/ds606/eval/refusal.py`
     - Start with regex patterns for typical refusal phrases.
     - Add minimal language coverage: English + common Hindi transliterations (e.g., "maaf", "main madad nahi kar sakta").
   - Add tests for true/false positives using small fixtures.

2. **Metrics**
   - Implement in `src/ds606/eval/metrics.py`:
     - `compute_asr(rows) -> float` (ASR definition should be consistent with your pipeline; stub if judging is pending)
     - `compute_refusal_rate(rows) -> float`
     - `cltg(asr_aligned_L, asr_aligned_en) -> float`
     - `ags(asr_base_L, asr_aligned_L, asr_base_en, asr_aligned_en) -> float`

3. **Bootstrap CI**
   - Implement `bootstrap_ci(values, n=1000, alpha=0.05, seed=0)` in `src/ds606/eval/stats.py` (create file).
   - Use it to report 95% CI for ASR, CLTG, AGS.

4. **Small “metrics runner” script**
   - Add `scripts/03_compute_metrics.py` that:
     - reads `outputs/generations/*.jsonl` (format you define)
     - writes `outputs/metrics/summary.json`

### Deliverables
- Refusal detector + tests
- Metric functions + bootstrap CI
- `scripts/03_compute_metrics.py` runnable on a small mocked generations file

---

## Notes (for both)
- Keep functions pure and deterministic.
- Prefer small PRs: schema+io first, then loader; refusal detector first, then metrics.
