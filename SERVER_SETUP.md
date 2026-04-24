# Server Setup Guide - Fix "ModuleNotFoundError: No module named 'ds606'"

## Problem

```
/usr/bin/python3: Error while finding module specification for 'ds606.cli'
(ModuleNotFoundError: No module named 'ds606')
```

## Solution: 2 Approaches

### ✅ Approach 1: Install Package (Recommended)

Run this **once** to install the package in development mode:

```bash
cd /workspace/DS606_project
python3 -m pip install -e .
```

**What this does:**
- Installs the `ds606` package in your Python environment
- Links to the source code in the repo
- Any changes to code are immediately reflected

**Then you can run:**
```bash
python3 -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --device-map cuda:3 \
  --max-samples 10
```

---

### Alternative Approach 2: Set PYTHONPATH

If you prefer not to install, set the environment variable:

```bash
export PYTHONPATH="/workspace/DS606_project/src:$PYTHONPATH"
python3 -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --device-map cuda:3
```

**Note:** You need to set `PYTHONPATH` every time you open a new terminal

---

## Step-by-Step Setup (Server)

### Step 1: SSH into Server

```bash
ssh user@your_server
```

### Step 2: Navigate to Project

```bash
cd /workspace/DS606_project
```

### Step 3: Install Package

```bash
python3 -m pip install -e .
```

**Example output:**
```
Successfully installed ds606-0.0.1
```

### Step 4: Verify Installation

```bash
python3 -c "from ds606.models.evaluate import evaluate_models_with_initial_response; print('✓ Success')"
```

Should print: `✓ Success`

### Step 5: Run Evaluation

```bash
# Quick test with 10 samples
python3 -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --device-map cuda:3 \
  --max-samples 10

# Or full evaluation
python3 -m ds606.cli evaluate-with-initial \
  --csv initial_malicious_final.csv \
  --device-map cuda:3
```

---

## If Install Fails

### Error: "No module named pip"

```bash
python3 -m ensurepip --upgrade
python3 -m pip install -e .
```

### Error: "Permission denied"

```bash
python3 -m pip install --user -e .
```

### Error: "Requires setup.py or pyproject.toml"

This means your project doesn't have installation metadata. Create `setup.py`:

```bash
cat > /workspace/DS606_project/setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="ds606",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
EOF

python3 -m pip install -e .
```

Or use `pyproject.toml` approach (shown below)

---

## Automated Setup Script

We created a setup script for you. On the server:

```bash
cd /workspace/DS606_project
bash setup_server.sh
```

This will:
1. Verify Python version
2. Install the package
3. Test the import
4. Show you the command to run

---

## Verify Everything Works

After setup, test with:

```bash
python3 -m ds606.cli evaluate-with-initial --help
```

Should show:
```
usage: python -m ds606.cli evaluate-with-initial [-h] --csv CSV ...
Evaluate models on prompts combined with initial malicious responses
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Install (one-time) | `python3 -m pip install -e .` |
| Test import | `python3 -c "from ds606.models.evaluate import evaluate_models_with_initial_response"` |
| Show help | `python3 -m ds606.cli evaluate-with-initial --help` |
| Run test | `python3 -m ds606.cli evaluate-with-initial --csv initial_malicious_final.csv --device-map cuda:3 --max-samples 10` |
| Run full | `python3 -m ds606.cli evaluate-with-initial --csv initial_malicious_final.csv --device-map cuda:3` |

---

## What's Being Installed?

The `pip install -e .` installs the `ds606` package from your local `src/` directory:

```
src/ds606/
├── __init__.py
├── cli.py              ← Contains evaluate-with-initial command
├── config.py
├── models/
│   └── evaluate.py     ← Contains evaluate_models_with_initial_response()
└── ...
```

Now Python can find these modules anywhere and use them with `from ds606.cli import main`

---

## Troubleshooting Summary

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'ds606'` | Run: `python3 -m pip install -e .` |
| `python: command not found` | Use: `python3` instead of `python` |
| `ERROR: Requires setup.py or pyproject.toml` | Create setup.py (shown above) |
| `Permission denied` | Run: `python3 -m pip install --user -e .` |
| Module still not found after install | Check: `python3 -c "import sys; print(sys.path)"` |

---

## After Installation

Your server setup is complete! Now you can:

1. **Run evaluations:**
   ```bash
   python3 -m ds606.cli evaluate-with-initial --csv initial_malicious_final.csv --device-map cuda:3
   ```

2. **Use in Python scripts:**
   ```python
   from ds606.models.evaluate import evaluate_models_with_initial_response
   evaluate_models_with_initial_response(...)
   ```

3. **Import any ds606 module:**
   ```python
   from ds606.cli import main
   from ds606.config import load_config_from_yaml
   ```

---

**Need help?** Check the error message against the troubleshooting table above, or see `START_HERE.md` for usage instructions.
