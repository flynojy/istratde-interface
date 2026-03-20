@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%runtime\python\python.exe"

if not exist "%PYTHON%" (
  echo Portable Python runtime not found: %PYTHON%
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

if "%ISTRATDE_BACKEND%"=="" set "ISTRATDE_BACKEND=torch"
if "%USE_PROCESS_POOL%"=="" set "USE_PROCESS_POOL=0"
if "%POP_SIZE%"=="" set "POP_SIZE=1000"
if "%MAX_FES%"=="" set "MAX_FES=1E6"
if "%CYCLE_NUM%"=="" set "CYCLE_NUM=1"
if "%FUN_ID_START%"=="" set "FUN_ID_START=15"
if "%FUN_ID_END%"=="" set "FUN_ID_END=15"
if "%VERBOSE_EVERY%"=="" set "VERBOSE_EVERY=1000"

echo Running MMES...
set "DEMO_OPTIMIZER=MMES"
"%PYTHON%" "%ROOT%test.py"
if errorlevel 1 exit /b 1

echo.
echo Running ISTRATDE...
set "DEMO_OPTIMIZER=ISTRATDE"
"%PYTHON%" "%ROOT%test.py"
if errorlevel 1 exit /b 1

echo.
echo Comparison summary:
"%PYTHON%" "%ROOT%compare_results.py"

endlocal
