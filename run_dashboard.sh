#!/bin/bash
# Launches the Streamlit dashboard with the project-local libomp.dylib on the
# search path, so XGBoost (if it's the champion model) loads correctly
# without a system-wide Homebrew install. See src/openmp_bootstrap.py.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/openmp:$DYLD_LIBRARY_PATH"
exec streamlit run app.py "$@"
