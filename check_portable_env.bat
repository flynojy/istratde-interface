@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%runtime\python\python.exe"

if not exist "%PYTHON%" (
  echo [FAIL] Portable Python runtime not found: %PYTHON%
  exit /b 1
)

set "TEMP=%ROOT%runtime\tmp"
set "TMP=%ROOT%runtime\tmp"
set "PIP_CACHE_DIR=%ROOT%runtime\pip-cache"
set "MPLCONFIGDIR=%ROOT%runtime\matplotlib"
set "PYTHONPYCACHEPREFIX=%ROOT%runtime\pycache"

if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"
if not exist "%PYTHONPYCACHEPREFIX%" mkdir "%PYTHONPYCACHEPREFIX%"

"%PYTHON%" "%ROOT%check_portable_env.py"

endlocal
