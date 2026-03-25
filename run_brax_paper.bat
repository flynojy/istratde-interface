@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%runtime\python\python.exe"

if not exist "%PYTHON%" (
  set "PYTHON=python"
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

if "%BRAX_ENVS%"=="" set "BRAX_ENVS=swimmer,hopper,reacher"
if "%BRAX_TIME_BUDGET_MINUTES%"=="" set "BRAX_TIME_BUDGET_MINUTES=60"
if "%BRAX_POP_SIZE%"=="" set "BRAX_POP_SIZE=10000"
if "%BRAX_MAX_EPISODE_LENGTH%"=="" set "BRAX_MAX_EPISODE_LENGTH=500"
if "%BRAX_NUM_EPISODES%"=="" set "BRAX_NUM_EPISODES=1"
if "%BRAX_HIDDEN_DIMS%"=="" set "BRAX_HIDDEN_DIMS=32,32"
if "%BRAX_LOG_INTERVAL_SECONDS%"=="" set "BRAX_LOG_INTERVAL_SECONDS=60"
if "%BRAX_SEED%"=="" set "BRAX_SEED=42"
if "%BRAX_SAVE_HTML%"=="" set "BRAX_SAVE_HTML=0"

"%PYTHON%" "%ROOT%brax_paper_benchmark.py"

endlocal
