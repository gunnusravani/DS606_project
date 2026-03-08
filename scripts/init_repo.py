from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DIRS = [
    "configs",
    "data/prompts",
    "data/raw",
    "data/processed",
    "notebooks",
    "outputs/generations",
    "outputs/metrics",
    "outputs/activations",
    "outputs/figures",
    "scripts",
    "src/ds606",
    "src/ds606/io",
    "src/ds606/data",
    "src/ds606/models",
    "src/ds606/eval",
    "src/ds606/mech",
    "src/ds606/analysis",
    "tests",
]

FILES = [
    "README.md",
    "requirements.txt",
    "configs/exp_default.yaml",
    "configs/models.yaml",
    "configs/datasets.yaml",
    "src/ds606/__init__.py",
    "src/ds606/cli.py",
    "src/ds606/config.py",
    "src/ds606/io/jsonl.py",
    "src/ds606/data/schema.py",
    "src/ds606/data/load.py",
    "src/ds606/eval/metrics.py",
    "src/ds606/eval/refusal.py",
    "JUNIORS_TASKS.md",
]

README = """# DS606: Cross-lingual safety alignment transfer

Repo skeleton for:
- Dataset loading/validation (JSONL)
- HF generation runner
- Safety eval + CLTG/AGS metrics
- (Later) mechanistic analysis hooks

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/init_repo.py
```
"""

REQUIREMENTS = """transformers>=4.41
torch
accelerate
pyyaml
tqdm
pandas
numpy
scipy
"""

EXP_DEFAULT = """seed: 0
output_dir: outputs

datasets:
  prompts_paths:
    - data/prompts/xsafety.jsonl
    - data/prompts/indicsafe.jsonl
    - data/prompts/india_harms.jsonl

model:
  name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
  torch_dtype: bfloat16
  device_map: auto

generation:
  max_new_tokens: 256
  temperature: 0.0
  top_p: 1.0
  do_sample: false
  batch_size: 4
"""

MODELS_YAML = """# Add base/aligned pairs here (populate later)
models: {}
"""

DATASETS_YAML = """# Dataset metadata (populate later)
datasets: {}
"""

JUNIORS_TASKS = """# Initial tasks for junior contributors

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
"""

def touch_gitkeep(dir_path: Path) -> None:
    gitkeep = dir_path / ".gitkeep"
    gitkeep.parent.mkdir(parents=True, exist_ok=True)
    gitkeep.touch(exist_ok=True)

def write_text_if_missing(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(content, encoding="utf-8")

def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")

    for rel in DIRS:
        p = PROJECT_ROOT / rel
        p.mkdir(parents=True, exist_ok=True)
        touch_gitkeep(p)

    for rel in FILES:
        p = PROJECT_ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)

    write_text_if_missing(PROJECT_ROOT / "README.md", README)
    write_text_if_missing(PROJECT_ROOT / "requirements.txt", REQUIREMENTS)
    write_text_if_missing(PROJECT_ROOT / "configs/exp_default.yaml", EXP_DEFAULT)
    write_text_if_missing(PROJECT_ROOT / "configs/models.yaml", MODELS_YAML)
    write_text_if_missing(PROJECT_ROOT / "configs/datasets.yaml", DATASETS_YAML)

    # Minimal module stubs so imports won’t fail as you implement
    write_text_if_missing(
        PROJECT_ROOT / "src/ds606/__init__.py",
        "__all__ = []\n",
    )
    write_text_if_missing(
        PROJECT_ROOT / "src/ds606/cli.py",
        "def main() -> None:\n"
        "    raise SystemExit('TODO: implement ds606.cli.main')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
    )
    write_text_if_missing(
        PROJECT_ROOT / "src/ds606/config.py",
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\n\n"
        "# TODO: define dataclasses for config + YAML loading\n",
    )
    write_text_if_missing(
        PROJECT_ROOT / "src/ds606/io/jsonl.py",
        "from __future__ import annotations\n\n"
        "# TODO: jsonl read/write helpers\n",
    )
    write_text_if_missing(
        PROJECT_ROOT / "src/ds606/data/schema.py",
        "from __future__ import annotations\n\n"
        "# TODO: prompt schema + validation\n",
    )
    write_text_if_missing(
        PROJECT_ROOT / "src/ds606/data/load.py",
        "from __future__ import annotations\n\n"
        "# TODO: load and merge datasets\n",
    )
    write_text_if_missing(
        PROJECT_ROOT / "src/ds606/eval/refusal.py",
        "from __future__ import annotations\n\n"
        "# TODO: refusal detector\n",
    )
    write_text_if_missing(
        PROJECT_ROOT / "src/ds606/eval/metrics.py",
        "from __future__ import annotations\n\n"
        "# TODO: ASR/refusal/CLTG/AGS computations\n",
    )
    write_text_if_missing(PROJECT_ROOT / "JUNIORS_TASKS.md", JUNIORS_TASKS)

    print("Done. Folder structure + starter files created.")

if __name__ == "__main__":
    main()
