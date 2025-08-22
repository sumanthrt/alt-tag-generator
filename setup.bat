@echo off
chcp 65001 >nul
echo ===========================
echo Setting up environment...
echo ===========================

REM Create venv if it doesn't exist
if not exist venv (
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

REM Detect NVIDIA GPU (nvidia-smi) and try CUDA 12.1 wheel if present
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo NVIDIA GPU detected. Trying CUDA 12.1 PyTorch wheel...
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo CUDA wheel install failed. Falling back to CPU wheel.
        pip install torch
    )
) else (
    echo No NVIDIA GPU detected. Installing CPU wheel...
    pip install torch
)

echo Installing project dependencies...
pip install transformers pillow tqdm

echo ===========================
echo Setup complete!
echo ===========================
echo To activate later: call venv\Scripts\activate
pause
