# Predictive Modeling and Risk Scoring for Bank Customer Churn

End-to-end churn intelligence system: data pipeline, model benchmarking, explainability, and an
interactive Streamlit dashboard.

## Project Structure

```
Bank churn/
├── European_Bank.csv          # Raw dataset (10,000 customers)
├── src/
│   ├── data_pipeline.py       # Cleaning, feature engineering, preprocessing pipeline
│   ├── train_models.py        # Trains & evaluates all models, saves the champion model
│   └── explainability.py      # Feature importance, SHAP, partial dependence plots
├── app.py                     # Streamlit dashboard
├── run_dashboard.sh            # Launches the dashboard (macOS/Linux) with libomp wired up
├── run_dashboard.bat           # Launches the dashboard (Windows)
├── models/                    # Saved pipeline + metadata (generated)
├── outputs/
│   ├── reports/                # Metrics CSVs, test predictions (generated)
│   └── figures/                 # ROC curves, confusion matrix, SHAP plots (generated)
├── research_paper.md          # Full EDA + methodology + findings write-up
├── executive_summary.md       # Non-technical summary for stakeholders
└── requirements.txt
```

## Setup

**macOS / Linux:**
```bash
cd "Bank churn"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell or cmd):**
```bat
cd "Bank churn"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### XGBoost's OpenMP dependency

XGBoost needs an OpenMP runtime at import time, and how that's satisfied differs by OS:

- **macOS**: the wheel looks for `libomp` at `/opt/homebrew/opt/libomp` (a Homebrew path). This
  dev machine has no Homebrew installed, and installing it requires sudo, so instead a copy of
  `libomp.dylib` (from conda-forge, arm64) is vendored at `.venv/lib/openmp/libomp.dylib` —
  project-local, no system/root changes. `src/openmp_bootstrap.py` points `DYLD_LIBRARY_PATH` at
  it, and is a no-op on other platforms. If you'd rather use a real Homebrew install instead of
  the vendored copy, `brew install libomp` and the standard rpath will take precedence
  automatically.
- **Windows**: the wheel depends on `vcomp140.dll`, which ships with the Microsoft Visual C++
  Redistributable — already present on most Windows machines (Python itself typically installs
  it). No vendoring is needed; `openmp_bootstrap.py`'s re-exec logic is skipped entirely on
  Windows. If XGBoost still fails to import, install the "Microsoft Visual C++ Redistributable for
  Visual Studio" (x64) from Microsoft and retry.
- **Linux**: install `libgomp` via your distro's package manager (e.g. `apt install libgomp1`) if
  it isn't already present.

## Running the Pipeline

```bash
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
cd src
python3 train_models.py       # trains all models (incl. XGBoost), saves the champion + metrics
python3 explainability.py     # generates SHAP / feature importance / PDP artifacts
```

## Running the Dashboard

**macOS / Linux:**
```bash
./run_dashboard.sh
```

**Windows:**
```bat
run_dashboard.bat
```

(Streamlit runs your script inside its own long-lived process via `exec()`, so the re-exec trick
`train_models.py` uses on macOS can't work there — `run_dashboard.sh` sets `DYLD_LIBRARY_PATH`
before Streamlit even starts. Running `streamlit run app.py` directly also works on macOS as long
as XGBoost isn't the champion model, or your shell already has `DYLD_LIBRARY_PATH` exported to
`.venv/lib/openmp`. On Windows, `streamlit run app.py` works directly with no extra setup.)

The dashboard has four modules: a customer risk calculator, a probability distribution view over
the held-out test set, a feature importance/SHAP explainability view, and a what-if scenario
simulator for testing retention interventions.
