# Quick Fix for Server - Copy & Paste This

## The Problem
```
ModuleNotFoundError: No module named 'ds606'
```

## The Solution - One Command

Run this **once** on your server:

```bash
cd /workspace/DS606_project && python3 -m pip install -e .
```

## That's it! Now you can run:

```bash
python3 -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --device-map cuda:3 \
  --max-samples 10
```

---

## If that doesn't work, try these alternatives:

### Alternative 1: Set PYTHONPATH

```bash
export PYTHONPATH="/workspace/DS606_project/src:$PYTHONPATH"
python3 -m ds606.cli evaluate-with-initial --csv initial_malicious_final.csv --device-map cuda:3
```

### Alternative 2: Run Python script instead

```bash
cd /workspace/DS606_project
python3 << 'EOF'
from src.ds606.models.evaluate import evaluate_models_with_initial_response

evaluate_models_with_initial_response(
    csv_path="initial_malicious_final.csv",
    device_map="cuda:3",
    output_path="outputs/evaluations_malicious/",
    max_samples=10,
)
EOF
```

### Alternative 3: Use the test script

```bash
cd /workspace/DS606_project
python3 scripts/test_evaluate_initial.py
```

---

## Detailed Setup Guide

See: `SERVER_SETUP.md` for full instructions with troubleshooting

## Quick Reference Table

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'ds606'` | `python3 -m pip install -e .` then re-run |
| `bash: python: command not found` | Use `python3` instead of `python` |
| Permission errors | Add `--user` flag: `python3 -m pip install --user -e .` |
| Still not working | Set PYTHONPATH: `export PYTHONPATH="/workspace/DS606_project/src:$PYTHONPATH"` |

---

## Run Evaluation

After fixing the import:

```bash
# Quick test (10 samples)
python3 -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --device-map cuda:3 \
  --max-samples 10

# Full evaluation
python3 -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --device-map cuda:3

# Using specific columns
python3 -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --device-map cuda:3 \
  --english-prompt-col question \
  --english-initial-col intital_malicious_english \
  --hindi-prompt-col hindi \
  --hindi-initial-col intital_malicious_hindi
```

Done! 🚀
