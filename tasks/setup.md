# Setup — Virtual Environment & Dependencies

**Status:** Not started  
**Do this before anything else.**

---

## What

Create a Python virtual environment, install all project dependencies into it, and verify the environment works. Nothing gets installed system-wide.

---

## Why a venv?

A virtual environment isolates your project's dependencies from the rest of your system. If you install XGBoost 2.1 here and another project needs XGBoost 1.7, they don't conflict. It also makes `requirements.txt` meaningful — you know exactly what this project needs, nothing more.

Rule: **always activate the venv before running any script or notebook in this project.**

---

## How to Set It Up

### Step 1 — Create the venv
From the project root:
```bash
python3 -m venv .venv
```

This creates a `.venv/` folder in the project root. It contains a self-contained Python installation.

### Step 2 — Activate it
```bash
# Linux / macOS / WSL
source .venv/bin/activate

# You'll see (.venv) at the start of your terminal prompt
```

To deactivate when you're done:
```bash
deactivate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Verify PySpark works (it needs Java)
```bash
python -c "from pyspark.sql import SparkSession; print('PySpark OK')"
```

If this fails with a Java error, install Java on WSL:
```bash
sudo apt-get update && sudo apt-get install -y default-jdk-headless
```

Then set `JAVA_HOME` if needed:
```bash
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
```

Add that export to your `~/.bashrc` so it persists across sessions.

### Step 5 — Verify MLflow
```bash
mlflow --version
```

### Step 6 — Add `.venv` to `.gitignore`
The venv must never be committed. Add this line to `.gitignore`:
```
.venv/
```

---

## Acceptance Criteria

- [ ] `.venv/` exists in the project root
- [ ] `source .venv/bin/activate` activates the environment
- [ ] `pip install -r requirements.txt` completes without errors
- [ ] `python -c "from pyspark.sql import SparkSession; print('PySpark OK')"` prints `PySpark OK`
- [ ] `mlflow --version` prints a version number
- [ ] `.venv/` is listed in `.gitignore`

---

## VS Code Integration

If you use VS Code, select the venv as your Python interpreter:

1. Open the command palette: `Ctrl+Shift+P`
2. Type: `Python: Select Interpreter`
3. Choose: `.venv/bin/python` (it should appear in the list automatically)

This makes sure VS Code uses the venv for linting, running cells in notebooks, and running scripts.
