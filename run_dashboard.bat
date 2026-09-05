@echo off
REM Launches the Streamlit dashboard on Windows.
REM Unlike macOS, XGBoost's Windows wheel resolves its OpenMP dependency
REM (vcomp140.dll) via the standard Windows DLL search path / the Microsoft
REM Visual C++ Redistributable, so no vendored-library workaround is needed
REM here (see src/openmp_bootstrap.py, which is a macOS-only no-op on
REM Windows). If XGBoost still fails to import, install the "Microsoft
REM Visual C++ Redistributable for Visual Studio" (x64) and retry.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py %*
