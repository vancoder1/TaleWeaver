@echo off
setlocal enabledelayedexpansion
if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit

:: Ensure Miniconda/Anaconda is installed and conda is in PATH
where conda >nul 2>nul
if errorlevel 1 (
    echo Conda is not installed or not found in PATH. Please install Miniconda/Anaconda and ensure conda is in your PATH.
    goto end
)

:: Define environment name and Python version
set ENV_NAME=taleweaver
set PYTHON_VERSION=3.12.4

:: Deactivate any active conda environment
call conda deactivate

:: Check if the environment already exists
call conda env list | findstr /C:"%ENV_NAME%" >nul
if %errorlevel% neq 0 (
    echo Creating conda environment %ENV_NAME% with Python %PYTHON_VERSION%
    call conda create -y -n %ENV_NAME% python=%PYTHON_VERSION%
) else (
    echo Conda environment %ENV_NAME% already exists.
)

:: Activate the conda environment
call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo Failed to activate conda environment %ENV_NAME%.
    goto end
)

:: Install requirements
if exist requirements.txt (
    echo Installing requirements from requirements.txt
    pip install -r requirements.txt
) else (
    echo requirements.txt not found.
    goto end
)

:: Run main.py
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