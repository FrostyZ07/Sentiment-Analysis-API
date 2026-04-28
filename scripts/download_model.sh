#!/usr/bin/env bash
# Download the fine-tuned model from HuggingFace Hub or W&B artefact.
# Usage: ./scripts/download_model.sh [hf|wandb]

set -euo pipefail

MODEL_DIR="${MODEL_PATH:-./models/distilbert-sentiment}"
SOURCE="${1:-hf}"

mkdir -p "$MODEL_DIR"

if [ "$SOURCE" = "hf" ]; then
    echo "Downloading model from HuggingFace Hub..."
    python -c "
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import os

model_id = os.environ.get('HF_MODEL_ID', 'FrostyZ07/distilbert-sentiment-amazon')
subfolder = os.environ.get('HF_SUBFOLDER', 'distilbert-sentiment')
save_path = os.environ.get('MODEL_PATH', './models/distilbert-sentiment')

print(f'Downloading {model_id} (subfolder={subfolder}) to {save_path}...')
try:
    tokenizer = DistilBertTokenizer.from_pretrained(model_id, subfolder=subfolder)
except Exception as e:
    print(f'Warning: Could not download tokenizer from {model_id}/{subfolder}. Falling back to base tokenizer. Error: {e}')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

try:
    model = DistilBertForSequenceClassification.from_pretrained(model_id, subfolder=subfolder)
except Exception as e:
    print(f'\nCRITICAL ERROR: Failed to download model from Hugging Face ({model_id}/{subfolder}).')
    print('1. Did you spell the HF_MODEL_ID exactly right?')
    print('2. If the repository is PRIVATE, you MUST add an HF_TOKEN variable in Railway Settings!')
    print(f'Detailed Error: {e}\n')
    exit(1)
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print('Download complete.')
"
elif [ "$SOURCE" = "wandb" ]; then
    echo "Downloading model from W&B artefact..."
    python -c "
import wandb, os, shutil

entity  = os.environ['WANDB_ENTITY']
project = os.environ['WANDB_PROJECT']
save_path = os.environ.get('MODEL_PATH', './models/distilbert-sentiment')

api = wandb.Api()
artifact = api.artifact(f'{entity}/{project}/distilbert-sentiment:latest')
artifact_dir = artifact.download()
shutil.copytree(artifact_dir, save_path, dirs_exist_ok=True)
print(f'Model downloaded to {save_path}')
"
else
    echo "Unknown source '$SOURCE'. Use 'hf' or 'wandb'."
    exit 1
fi

echo "Model ready at: $MODEL_DIR"
ls -la "$MODEL_DIR"
