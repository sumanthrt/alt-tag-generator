@echo off
chcp 65001 >nul
echo ===========================
echo Setting up environment...
echo ===========================

echo Installing required packages globally...
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow tqdm

echo Setup complete!
pause
