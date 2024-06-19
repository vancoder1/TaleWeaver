@echo off
setlocal enabledelayedexpansion
if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit

:: Ensure Miniconda/Anaconda is installed and conda is in PATH
where conda >nul 2>nul
if errorlevel 1 (
    echo Conda is not installed or not found in PATH. Please install Miniconda/Anaconda and ensure conda is in your PATH.
    goto end
)

:: Define environment name
set ENV_NAME=taleweaver

:: Deactivate any active conda environment
call conda deactivate

:: Activate the conda environment
call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo Failed to activate conda environment %ENV_NAME%.
    goto end
)

:: Run server.py
if exist server.py (
    echo Running server.py
    python server.py
) else (
    echo server.py not found.
    goto end
)

:: Deactivate conda environment before exit
call conda deactivate

:end
pause
endlocal