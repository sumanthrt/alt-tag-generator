#!/bin/bash

MODEL_DIR="./blip2-opt-2.7b"

echo "🟢 Setting up Alt Tag Generator..."

# Check if BLIP2 model folder exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "⚠️ BLIP2 model folder not found. Downloading model locally..."
    python3 - <<END
from transformers import Blip2Processor, Blip2ForConditionalGeneration
MODEL_DIR = "$MODEL_DIR"
print("Downloading BLIP2 processor...")
Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=MODEL_DIR)
print("Downloading BLIP2 model...")
Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=MODEL_DIR)
print("✅ Model downloaded successfully!")
END
else
    echo "✅ BLIP2 model folder already exists."
fi

# Install Python dependencies including accelerate
echo "Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install torch transformers Pillow tqdm accelerate

echo "✅ Setup complete. You can now run the generator using ./run.sh"
