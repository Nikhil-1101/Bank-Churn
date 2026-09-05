"""Makes the project-local libomp.dylib visible to XGBoost on macOS.

XGBoost's macOS wheel hard-codes an rpath of /opt/homebrew/opt/libomp/lib and
this machine has no Homebrew installed. Rather than requiring a system-wide,
sudo-gated Homebrew install, we ship libomp.dylib inside the project's own
virtualenv (.venv/lib/openmp/) and make it discoverable via
DYLD_LIBRARY_PATH. dyld only reads that variable at process launch, so if
it isn't already set we re-exec the current interpreter once with it added.
Call ensure_openmp() before importing xgboost (directly or via joblib.load).
"""

import os
import sys


def ensure_openmp() -> None:
    if sys.platform != "darwin":
        return

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_dir = os.path.join(project_root, ".venv", "lib", "openmp")
    lib_file = os.path.join(lib_dir, "libomp.dylib")
    if not os.path.exists(lib_file):
        return

    current = os.environ.get("DYLD_LIBRARY_PATH", "")
    if lib_dir in current.split(os.pathsep):
        return  # already active in this process (post re-exec)

    os.environ["DYLD_LIBRARY_PATH"] = (
        lib_dir + (os.pathsep + current if current else "")
    )
    os.execv(sys.executable, [sys.executable] + sys.argv)
