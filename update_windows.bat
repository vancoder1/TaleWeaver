@echo off
setlocal enabledelayedexpansion
if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit

:: Ensure Git is installed and in PATH
where git >nul 2>nul
if errorlevel 1 (
    echo Git is not installed or not found in PATH. Please install Git and ensure it's in your PATH.
    goto end
)

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

:: Check if the environment exists, if not create it
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

:: Check for Git updates
echo Checking for updates...
git fetch origin
git status -uno | findstr "Your branch is up to date" >nul
if %errorlevel% neq 0 (
    echo Updates available. Pulling changes...
    git pull origin main
    
    :: Reinstall requirements in case they've changed
    if exist requirements.txt (
        echo Updating requirements...
        pip install -r requirements.txt
    )
    
    echo Update complete.
) else (
    echo Your project is up to date.
)

:: Deactivate conda environment before exit
call conda deactivate

:end
pause
endlocal