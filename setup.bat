@echo off
set MODEL_DIR=blip2-opt-2.7b

echo 🟢 Setting up Alt Tag Generator...

:: Check if model folder exists
if not exist "%MODEL_DIR%" (
    echo ⚠️ BLIP2 model folder not found. Downloading model locally...
    python - <<END
from transformers import Blip2Processor, Blip2ForConditionalGeneration
MODEL_DIR = r"%MODEL_DIR%"
print("Downloading BLIP2 processor...")
Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=MODEL_DIR)
print("Downloading BLIP2 model...")
Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=MODEL_DIR)
print("✅ Model downloaded successfully!")
END
) else (
    echo ✅ BLIP2 model folder already exists.
)

:: Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install torch transformers Pillow tqdm accelerate

echo ✅ Setup complete. You can now run the generator using run.bat
pause
