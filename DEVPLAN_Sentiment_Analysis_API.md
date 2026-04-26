# End-to-End Development Plan
## Sentiment Analysis API — DistilBERT on Amazon Reviews

**Based on:** PRD v1.0  
**Plan Version:** 1.0  
**Date:** April 2026  
**Total Estimated Effort:** 8 Weeks  

---

## How to Use This Document

This plan is a step-by-step build guide. Every section maps to a PRD requirement. Each phase contains:
- Exact commands to run
- Complete file contents to create
- Acceptance checks before moving forward
- Common pitfalls and how to avoid them

Work through phases in order. Do not skip ahead — later phases depend on earlier ones.

---

## Table of Contents

- [Phase 0 — Project Scaffold & Tooling](#phase-0--project-scaffold--tooling)
- [Phase 1 — Data Exploration & EDA](#phase-1--data-exploration--eda)
- [Phase 2 — Model Fine-Tuning with W&B Logging](#phase-2--model-fine-tuning-with-wb-logging)
- [Phase 3 — FastAPI Application Core](#phase-3--fastapi-application-core)
- [Phase 4 — API Endpoints, Middleware & Docs](#phase-4--api-endpoints-middleware--docs)
- [Phase 5 — Testing Suite](#phase-5--testing-suite)
- [Phase 6 — Dockerisation](#phase-6--dockerisation)
- [Phase 7 — CI/CD with GitHub Actions](#phase-7--cicd-with-github-actions)
- [Phase 8 — Deployment to Railway](#phase-8--deployment-to-railway)
- [Phase 9 — Stand-Out Features](#phase-9--stand-out-features)
- [Phase 10 — Polish, Load Testing & README](#phase-10--polish-load-testing--readme)
- [Appendix A — Full File Tree](#appendix-a--full-file-tree)
- [Appendix B — Environment Variables Reference](#appendix-b--environment-variables-reference)
- [Appendix C — Troubleshooting](#appendix-c--troubleshooting)

---

## Phase 0 — Project Scaffold & Tooling

**PRD Sections:** 8.1, 8.2, 12.4  
  
**Goal:** Reproducible local environment and repository skeleton before writing a single line of ML code.

---

### 0.1 Create the GitHub Repository

1. Go to github.com → New Repository
2. Name: `sentiment-analysis-api`
3. Visibility: **Public** (required for free GitHub Actions minutes and portfolio visibility)
4. Initialize with `README.md`, `.gitignore` (Python template), MIT License
5. Clone locally:

```bash
git clone https://github.com/<your-username>/sentiment-analysis-api.git
cd sentiment-analysis-api
```

---

### 0.2 Create the Full Directory Structure

Run this from the project root to create all directories and placeholder files at once:

```bash
mkdir -p .github/workflows
mkdir -p app/api/v1/routes
mkdir -p app/core
mkdir -p app/schemas
mkdir -p app/services
mkdir -p notebooks
mkdir -p data
mkdir -p models
mkdir -p tests
mkdir -p scripts
mkdir -p docker
mkdir -p sweeps
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/prometheus

# Placeholder files so git tracks empty dirs
touch models/.gitkeep
touch app/api/__init__.py
touch app/api/v1/__init__.py
touch app/api/v1/routes/__init__.py
touch app/core/__init__.py
touch app/schemas/__init__.py
touch app/services/__init__.py
```

---

### 0.3 Pin All Dependency Versions

Create three requirements files. Pinning versions prevents "it worked last week" breakage.

**`requirements.txt`** — API runtime only:

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.1
pydantic-settings==2.3.0
transformers==4.40.2
torch==2.2.2
slowapi==0.1.9
python-multipart==0.0.9
httpx==0.27.0
structlog==24.1.0
python-dotenv==1.0.1
```

**`requirements-train.txt`** — Training environment only:

```
transformers==4.40.2
datasets==2.18.0
torch==2.2.2
wandb==0.17.0
scikit-learn==1.4.2
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
seaborn==0.13.2
accelerate==0.30.1
evaluate==0.4.2
```

**`requirements-dev.txt`** — Development and CI tools:

```
pytest==8.2.0
pytest-asyncio==0.23.6
pytest-cov==5.0.0
httpx==0.27.0
ruff==0.4.4
black==24.4.2
isort==5.13.2
locust==2.29.0
```

**`requirements-standout.txt`** — Stand-out features (Phase 9):

```
evidently==0.4.33
prometheus-fastapi-instrumentator==7.0.0
apscheduler==3.10.4
optimum==1.20.0
aiosqlite==0.20.0
```

---

### 0.4 Python Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

### 0.5 Code Quality Configuration

**`pyproject.toml`** — single config file for all formatters and linters:

```toml
[tool.black]
line-length = 88
target-version = ["py311"]
include = '\.pyi?$'
exclude = '''
/(
    \.git
  | \.venv
  | notebooks
)/
'''

[tool.isort]
profile = "black"
line_length = 88

[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "W", "F", "I"]
ignore = ["E501"]
exclude = [".venv", "notebooks"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "--cov=app --cov-report=term-missing --cov-fail-under=80"
```

---

### 0.6 Environment Variables Template

**`.env.example`** (commit this — it has NO secrets):

```dotenv
# === Model Configuration ===
# Local path OR HuggingFace Hub model ID
MODEL_PATH=./models/distilbert-sentiment

# === Server ===
PORT=8000
LOG_LEVEL=INFO

# === CORS ===
# Comma-separated list. Use * for development only.
ALLOWED_ORIGINS=*

# === Rate Limiting ===
RATE_LIMIT_PER_MINUTE=60

# === Experiment Tracking ===
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=sentiment-analysis-distilbert
WANDB_ENTITY=your_wandb_username

# === HuggingFace ===
# Required only if pushing model to HF Hub
HF_TOKEN=your_hf_token_here
HF_PUSH=false

# === Stand-Out: API Keys (comma-separated SHA-256 hashes) ===
API_KEYS=

# === Stand-Out: Second model for A/B testing (optional) ===
MODEL_PATH_V2=
```

Copy to `.env` (never commit `.env`):

```bash
cp .env.example .env
```

**`.gitignore`** additions (append to the Python default):

```gitignore
# Project specific
.env
models/distilbert-sentiment/
*.pt
*.bin
*.safetensors
wandb/
reports/
__pycache__/
.pytest_cache/
htmlcov/
.coverage
dist/
*.egg-info/
```

---

### 0.7 Phase 0 Acceptance Check

```bash
# Lint passes on empty scaffold
ruff check app/
black --check app/

# Git is clean
git add -A
git commit -m "chore: initial project scaffold and tooling setup"
git push
```

✅ Move to Phase 1 when: repository is on GitHub, directory structure exists, all three requirements files committed, pyproject.toml committed, .env.example committed.

---

## Phase 1 — Data Exploration & EDA

**PRD Sections:** 9.1–9.4, 6.2 Section 1  

**Goal:** Understand the dataset deeply before touching the model. EDA shapes preprocessing decisions.  
**Environment:** Google Colab with T4 GPU (free) or local with CPU (EDA only)

---

### 1.1 Install Training Dependencies in Colab

Add this as the first cell of your notebook:

```python
# Cell 0: Install dependencies
!pip install transformers==4.40.2 datasets==2.18.0 wandb==0.17.0 \
    scikit-learn==1.4.2 evaluate==0.4.2 accelerate==0.30.1 -q
```

---

### 1.2 Environment & Seed Setup

```python
# Cell 1: Imports and reproducibility
import os
import random
import numpy as np
import torch
import wandb
from datasets import load_dataset
from transformers import DistilBertTokenizer, set_seed

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# ── Device ───────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ── Constants ────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
TRAIN_SAMPLES = 200_000
VAL_SAMPLES   = 25_000
TEST_SAMPLES  = 25_000
LABEL_NAMES   = ["negative", "positive"]
```

---

### 1.3 Initialize W&B Run

```python
# Cell 2: W&B initialization
import wandb

# Login (paste API key when prompted, or set WANDB_API_KEY env var)
wandb.login()

run = wandb.init(
    project="sentiment-analysis-distilbert",
    name="eda-and-preprocessing",
    tags=["eda", "phase-1"],
    config={
        "seed": SEED,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "train_samples": TRAIN_SAMPLES,
        "val_samples": VAL_SAMPLES,
        "test_samples": TEST_SAMPLES,
        "dataset": "amazon_polarity",
    }
)
print(f"W&B Run URL: {run.get_url()}")
```

---

### 1.4 Load Dataset and EDA

```python
# Cell 3: Load amazon_polarity dataset
dataset = load_dataset("amazon_polarity")

print("Dataset structure:")
print(dataset)
print("\nFeatures:", dataset["train"].features)
print("\nTrain size:", len(dataset["train"]))
print("Test size:", len(dataset["test"]))
```

```python
# Cell 4: Class distribution analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = dataset["train"].to_pandas()

# Class counts
label_counts = train_df["label"].value_counts().reset_index()
label_counts.columns = ["label", "count"]
label_counts["label_name"] = label_counts["label"].map({0: "negative", 1: "positive"})

print("Class Distribution (Full Train Set):")
print(label_counts)
print(f"\nClass balance ratio: {label_counts['count'].min() / label_counts['count'].max():.4f}")

# Log to W&B as bar chart
wandb.log({
    "class_distribution": wandb.plot.bar(
        wandb.Table(dataframe=label_counts[["label_name", "count"]]),
        "label_name",
        "count",
        title="Class Distribution — Full Training Set"
    )
})
```

```python
# Cell 5: Text length distribution
train_df["text_length"] = train_df["content"].str.len()
train_df["word_count"] = train_df["content"].str.split().str.len()

print("Text length statistics (characters):")
print(train_df["text_length"].describe())

print("\nWord count statistics:")
print(train_df["word_count"].describe())

# What % of reviews exceed 512 tokens? (important for truncation decision)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
sample_500 = train_df.sample(500, random_state=SEED)
token_lengths = sample_500["content"].apply(
    lambda x: len(tokenizer.encode(x, add_special_tokens=True))
)
pct_over_128 = (token_lengths > 128).mean() * 100
pct_over_256 = (token_lengths > 256).mean() * 100
pct_over_512 = (token_lengths > 512).mean() * 100

print(f"\nToken length analysis (sample of 500):")
print(f"  > 128 tokens: {pct_over_128:.1f}%")
print(f"  > 256 tokens: {pct_over_256:.1f}%")
print(f"  > 512 tokens: {pct_over_512:.1f}%")

# Log histogram to W&B
wandb.log({
    "token_length_distribution": wandb.plot.histogram(
        wandb.Table(data=[[l] for l in token_lengths], columns=["token_length"]),
        "token_length",
        title="Token Length Distribution (sample n=500)"
    ),
    "pct_reviews_over_128_tokens": pct_over_128,
    "pct_reviews_over_512_tokens": pct_over_512,
})
```

```python
# Cell 6: Sample reviews per class
print("=" * 60)
print("NEGATIVE REVIEWS (label=0) — Sample:")
print("=" * 60)
neg_samples = train_df[train_df["label"] == 0]["content"].sample(3, random_state=SEED)
for i, text in enumerate(neg_samples):
    print(f"\n[{i+1}] {text[:300]}...")

print("\n" + "=" * 60)
print("POSITIVE REVIEWS (label=1) — Sample:")
print("=" * 60)
pos_samples = train_df[train_df["label"] == 1]["content"].sample(3, random_state=SEED)
for i, text in enumerate(pos_samples):
    print(f"\n[{i+1}] {text[:300]}...")
```

---

### 1.5 Data Cleaning Function

```python
# Cell 7: Preprocessing utilities
import re

def clean_text(text: str) -> str:
    """
    Clean review text before tokenisation.
    - Strip HTML tags
    - Normalize whitespace
    - Remove reviews shorter than 3 words (handled in filter step)
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid_review(text: str, min_words: int = 3) -> bool:
    """Filter out reviews too short to carry signal."""
    return len(text.split()) >= min_words

# Test the functions
test_text = "<b>Great product!</b>  Really    loved it."
print(f"Original:  '{test_text}'")
print(f"Cleaned:   '{clean_text(test_text)}'")
print(f"Valid:     {is_valid_review(clean_text(test_text))}")
```

---

### 1.6 Create Stratified Train/Val/Test Splits

```python
# Cell 8: Build sampled, stratified splits
from datasets import DatasetDict
import datasets as hf_datasets

def prepare_splits(dataset, train_n, val_n, test_n, seed):
    """
    Sample stratified subsets from amazon_polarity.
    Returns a DatasetDict with train/validation/test splits.
    """
    # Apply cleaning to the full dataset lazily
    def preprocess(example):
        example["content"] = clean_text(example["content"])
        return example

    dataset = dataset.map(preprocess, num_proc=2)

    # Filter very short reviews
    dataset = dataset.filter(lambda x: is_valid_review(x["content"]))

    # Sample training set (stratified by label)
    full_train = dataset["train"]
    # Split into pos/neg for stratified sampling
    neg_train = full_train.filter(lambda x: x["label"] == 0)
    pos_train = full_train.filter(lambda x: x["label"] == 1)

    # Each class gets half the requested samples
    neg_train_sample = neg_train.shuffle(seed=seed).select(range(train_n // 2))
    pos_train_sample = pos_train.shuffle(seed=seed).select(range(train_n // 2))
    train_split = hf_datasets.concatenate_datasets([neg_train_sample, pos_train_sample])
    train_split = train_split.shuffle(seed=seed)

    # Sample val/test from original test set (different source than train)
    full_test = dataset["test"]
    neg_test = full_test.filter(lambda x: x["label"] == 0)
    pos_test = full_test.filter(lambda x: x["label"] == 1)

    neg_val = neg_test.shuffle(seed=seed).select(range(val_n // 2))
    pos_val = pos_test.shuffle(seed=seed).select(range(val_n // 2))
    val_split = hf_datasets.concatenate_datasets([neg_val, pos_val]).shuffle(seed=seed)

    neg_test_final = neg_test.shuffle(seed=seed+1).select(range(val_n, val_n + test_n // 2))
    pos_test_final = pos_test.shuffle(seed=seed+1).select(range(val_n, val_n + test_n // 2))
    test_split = hf_datasets.concatenate_datasets([neg_test_final, pos_test_final]).shuffle(seed=seed)

    return DatasetDict({
        "train": train_split,
        "validation": val_split,
        "test": test_split,
    })

splits = prepare_splits(dataset, TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES, SEED)
print("Split sizes:")
for name, ds in splits.items():
    neg = sum(1 for x in ds if x["label"] == 0)
    pos = sum(1 for x in ds if x["label"] == 1)
    print(f"  {name}: {len(ds)} (neg={neg}, pos={pos})")

# Log split stats to W&B
wandb.config.update({
    "actual_train_size": len(splits["train"]),
    "actual_val_size": len(splits["validation"]),
    "actual_test_size": len(splits["test"]),
})
```

---

### 1.7 Phase 1 Acceptance Check

Before moving on, verify:

- [ ] W&B EDA run is visible in your project dashboard
- [ ] Class distribution bar chart logged to W&B
- [ ] Token length histogram logged to W&B
- [ ] `pct_reviews_over_128_tokens` is < 30% (justifies `max_length=128`)
- [ ] `splits` DatasetDict contains 3 balanced splits
- [ ] `clean_text()` and `is_valid_review()` functions work on edge cases

---

## Phase 2 — Model Fine-Tuning with W&B Logging

**PRD Sections:** 6.2 Sections 3–8, 10.1–10.4  


**Goal:** Fine-tune DistilBERT, hit ≥ 92% accuracy, save model artefact.  
**Environment:** Google Colab T4 GPU (free tier, ~2–3 hours training time for 200k samples)

---

### 2.1 Tokenization

```python
# Cell 9: Tokenize all splits
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    return tokenizer(
        batch["content"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_attention_mask=True,
    )

tokenized = splits.map(
    tokenize_batch,
    batched=True,
    batch_size=1000,
    remove_columns=["content", "title"],  # keep only model inputs + label
    desc="Tokenizing",
)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

print("Tokenized dataset:")
print(tokenized)
print("\nSample tensor shapes:")
sample = tokenized["train"][0]
for k, v in sample.items():
    print(f"  {k}: {v.shape}")
```

---

### 2.2 Metrics Function

```python
# Cell 10: Metrics for Trainer
import evaluate
import numpy as np

accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_macro = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    f1_weighted = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "f1_macro": f1_macro["f1"],
        "f1_weighted": f1_weighted["f1"],
    }
```

---

### 2.3 Model Initialization

```python
# Cell 11: Initialize model
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
)
model.to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

wandb.config.update({
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
})
```

---

### 2.4 Training Arguments

```python
# Cell 12: Training configuration
from transformers import TrainingArguments

# ── Hyperparameters (all logged to W&B via report_to="wandb") ────────
LEARNING_RATE    = 2e-5
NUM_EPOCHS       = 3
BATCH_SIZE       = 16
WARMUP_RATIO     = 0.1
WEIGHT_DECAY     = 0.01
OUTPUT_DIR       = "./checkpoints/distilbert-sentiment"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=100,
    report_to="wandb",          # ← All metrics auto-logged to W&B
    run_name=wandb.run.name,
    fp16=torch.cuda.is_available(),  # Mixed precision on GPU
    dataloader_num_workers=2,
    save_total_limit=2,          # Keep only 2 checkpoints to save disk
    seed=SEED,
)

wandb.config.update({
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "warmup_ratio": WARMUP_RATIO,
    "weight_decay": WEIGHT_DECAY,
    "fp16": torch.cuda.is_available(),
})
```

---

### 2.5 Training Execution

```python
# Cell 13: Train the model
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

print("Starting fine-tuning...")
print(f"Training samples: {len(tokenized['train'])}")
print(f"Steps per epoch:  {len(tokenized['train']) // BATCH_SIZE}")
print(f"Total steps:      {len(tokenized['train']) // BATCH_SIZE * NUM_EPOCHS}")

train_result = trainer.train()

print("\nTraining complete!")
print(f"Training loss:    {train_result.training_loss:.4f}")
print(f"Training runtime: {train_result.metrics['train_runtime']:.0f}s")
```

---

### 2.6 Test Set Evaluation & W&B Logging

```python
# Cell 14: Full test set evaluation
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# Run inference on test set
test_output = trainer.predict(tokenized["test"])
test_logits = test_output.predictions
test_labels = test_output.label_ids
test_preds  = np.argmax(test_logits, axis=-1)
test_probs  = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()

# ── Classification Report ─────────────────────────────────────────────
print("Test Set Classification Report:")
print(classification_report(test_labels, test_preds, target_names=LABEL_NAMES))

# ── ROC-AUC ──────────────────────────────────────────────────────────
roc_auc = roc_auc_score(test_labels, test_probs[:, 1])
print(f"ROC-AUC: {roc_auc:.4f}")

# ── Log confusion matrix to W&B ───────────────────────────────────────
wandb.log({
    "test/confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=test_labels.tolist(),
        preds=test_preds.tolist(),
        class_names=LABEL_NAMES,
    ),
    "test/roc_auc": roc_auc,
    "test/accuracy": (test_preds == test_labels).mean(),
})

# ── Log error analysis table to W&B ──────────────────────────────────
# Find misclassified examples for inspection
wrong_mask = test_preds != test_labels
wrong_indices = np.where(wrong_mask)[0][:50]  # First 50 errors

test_texts = [splits["test"][int(i)]["content"] for i in wrong_indices]
error_table = wandb.Table(columns=["text", "true_label", "predicted", "confidence"])
for idx in wrong_indices:
    error_table.add_data(
        test_texts[list(wrong_indices).index(idx)][:200],
        LABEL_NAMES[test_labels[idx]],
        LABEL_NAMES[test_preds[idx]],
        float(test_probs[idx].max()),
    )
wandb.log({"test/error_analysis": error_table})

# ── Assert minimum thresholds before saving ───────────────────────────
test_accuracy = (test_preds == test_labels).mean()
from sklearn.metrics import f1_score
test_f1 = f1_score(test_labels, test_preds, average="macro")

print(f"\n{'='*50}")
print(f"FINAL METRICS")
print(f"{'='*50}")
print(f"Accuracy: {test_accuracy:.4f} (required: ≥ 0.92)")
print(f"F1 Macro: {test_f1:.4f}  (required: ≥ 0.91)")

assert test_accuracy >= 0.92, f"❌ Accuracy {test_accuracy:.4f} below threshold 0.92"
assert test_f1 >= 0.91,       f"❌ F1 {test_f1:.4f} below threshold 0.91"
print("\n✅ All thresholds met. Proceeding to model save.")
```

---

### 2.7 Save Model and Log as W&B Artefact

```python
# Cell 15: Save model + tokenizer
MODEL_SAVE_PATH = "./models/distilbert-sentiment"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

# Save training args for reproducibility
import json
training_args_dict = training_args.to_dict()
with open(f"{MODEL_SAVE_PATH}/training_args.json", "w") as f:
    json.dump(training_args_dict, f, indent=2, default=str)

print(f"Model saved to: {MODEL_SAVE_PATH}")
print("Files:")
for f in os.listdir(MODEL_SAVE_PATH):
    size_mb = os.path.getsize(f"{MODEL_SAVE_PATH}/{f}") / 1e6
    print(f"  {f}: {size_mb:.1f} MB")
```

```python
# Cell 16: Write model card
model_card = f"""# DistilBERT Sentiment Analysis — Amazon Reviews

## Model Description
Fine-tuned `distilbert-base-uncased` for binary sentiment classification
on Amazon product reviews (positive / negative).

## Training Data
- **Dataset:** amazon_polarity (HuggingFace)
- **Train samples:** {len(splits['train']):,}
- **Validation samples:** {len(splits['validation']):,}
- **Test samples:** {len(splits['test']):,}

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | {LEARNING_RATE} |
| Epochs | {NUM_EPOCHS} |
| Batch Size | {BATCH_SIZE} |
| Warmup Ratio | {WARMUP_RATIO} |
| Weight Decay | {WEIGHT_DECAY} |
| Max Token Length | {MAX_LENGTH} |

## Performance (Test Set)
| Metric | Value |
|--------|-------|
| Accuracy | {test_accuracy:.4f} |
| F1 Macro | {test_f1:.4f} |
| ROC-AUC | {roc_auc:.4f} |

## W&B Run
{wandb.run.get_url()}

## Labels
- `0` → negative
- `1` → positive

## Usage
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="./models/distilbert-sentiment")
result = classifier("This product is amazing!")
print(result)  # [{{'label': 'positive', 'score': 0.98}}]
```
"""

with open(f"{MODEL_SAVE_PATH}/model_card.md", "w") as f:
    f.write(model_card)

with open("model_card.md", "w") as f:
    f.write(model_card)

print("Model card written.")
```

```python
# Cell 17: Log model as W&B Artefact
artifact = wandb.Artifact(
    name="distilbert-sentiment",
    type="model",
    description="Fine-tuned DistilBERT for Amazon review sentiment classification",
    metadata={
        "test_accuracy": float(test_accuracy),
        "test_f1_macro": float(test_f1),
        "roc_auc": float(roc_auc),
        "base_model": MODEL_NAME,
        "dataset": "amazon_polarity",
        "max_length": MAX_LENGTH,
        "train_samples": len(splits["train"]),
    }
)
artifact.add_dir(MODEL_SAVE_PATH)
wandb.log_artifact(artifact, aliases=["latest", "v1"])

wandb.finish()
print("✅ W&B run complete. Model artefact logged.")
print(f"Artefact URL: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/artifacts/model/distilbert-sentiment")
```

---

### 2.8 Inference Demo

```python
# Cell 18: Verify inference before deploying
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model=MODEL_SAVE_PATH,
    tokenizer=MODEL_SAVE_PATH,
    device=0 if torch.cuda.is_available() else -1,
)

test_reviews = [
    "Absolutely love this product! Best purchase I've made all year.",
    "Complete waste of money. Broke after two days of light use.",
    "It's okay. Nothing special but gets the job done.",
    "The quality is outstanding and shipping was super fast!",
    "Terrible customer service and the item looks nothing like the photos.",
]

print("Inference Demo:")
print("=" * 60)
for review in test_reviews:
    result = classifier(review)[0]
    confidence = result["score"]
    label = result["label"].lower()
    bar = "█" * int(confidence * 20)
    print(f"\n[{label.upper():8s}] {confidence:.3f} {bar}")
    print(f"  \"{review[:70]}...\"" if len(review) > 70 else f"  \"{review}\"")
```

---

### 2.9 Download Model from Colab

In Colab, zip and download the model:

```python
# Cell 19: Zip for download
import shutil
shutil.make_archive("distilbert-sentiment", "zip", ".", "models/distilbert-sentiment")
print("Created distilbert-sentiment.zip — download via Files panel →")
```

Then in your local repo:

```bash
# Unzip into the models directory
unzip distilbert-sentiment.zip -d .
# Verify
ls models/distilbert-sentiment/
# Should show: config.json, pytorch_model.bin (or model.safetensors), tokenizer files, model_card.md
```

---

### 2.10 Phase 2 Acceptance Check

- [ ] Training notebook runs end-to-end without errors
- [ ] Test accuracy ≥ 92%, F1 macro ≥ 0.91 (assert passes)
- [ ] W&B run publicly visible with confusion matrix, loss curves, error table
- [ ] Model files present in `models/distilbert-sentiment/`
- [ ] `model_card.md` exists in project root with correct metrics
- [ ] W&B artefact tagged `latest` and `v1`

---

## Phase 3 — FastAPI Application Core

**PRD Sections:** 7.1–7.5  

**Goal:** Working API that loads the model and serves single predictions.

---

### 3.1 Application Settings

**`app/core/config.py`**:

```python
"""Application configuration via environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """All settings loaded from environment / .env file."""

    # Model
    model_path: str = "./models/distilbert-sentiment"
    model_path_v2: str = ""  # Optional second model for A/B testing

    # Server
    port: int = 8000
    log_level: str = "INFO"
    allowed_origins: str = "*"

    # Rate limiting
    rate_limit_per_minute: int = 60

    # API Keys (comma-separated SHA-256 hashes; empty = no auth required)
    api_keys: str = ""

    # W&B (used for model metadata endpoint)
    wandb_project: str = "sentiment-analysis-distilbert"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def api_keys_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
```

---

### 3.2 Model Loader

**`app/core/model.py`**:

```python
"""Model loading and management."""
import logging
import time
from pathlib import Path
from dataclasses import dataclass

import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Holds model, tokenizer and metadata together."""
    model: DistilBertForSequenceClassification
    tokenizer: DistilBertTokenizer
    model_name: str
    model_path: str
    device: torch.device
    labels: dict[int, str]


def load_model(model_path: str, version_tag: str = "v1") -> ModelBundle:
    """
    Load fine-tuned DistilBERT model from local path or HuggingFace Hub.

    Args:
        model_path: Local directory path or HF Hub model ID.
        version_tag: Label for this model version (e.g., "v1", "v2").

    Returns:
        ModelBundle with loaded model and tokenizer.

    Raises:
        RuntimeError: If model files are not found.
    """
    start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from '{model_path}' on {device}...")

    # Validate local path exists
    path = Path(model_path)
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"MODEL_PATH '{model_path}' exists but is not a directory.")
    if path.exists() and not (path / "config.json").exists():
        raise RuntimeError(f"No config.json found in '{model_path}'. Is this a valid model directory?")

    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{model_path}': {e}") from e

    elapsed = time.perf_counter() - start
    labels = {int(k): v for k, v in model.config.id2label.items()}

    logger.info(
        f"Model loaded successfully in {elapsed:.2f}s "
        f"| version={version_tag} | labels={labels} | device={device}"
    )

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_name=f"distilbert-sentiment-{version_tag}",
        model_path=model_path,
        device=device,
        labels=labels,
    )
```

---

### 3.3 Request & Response Schemas

**`app/schemas/request.py`**:

```python
"""Pydantic request schemas."""
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Single review prediction request."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Product review text to classify.",
        json_schema_extra={"example": "This product exceeded all my expectations!"},
    )
    return_probabilities: bool = Field(
        default=False,
        description="If true, include per-class probability scores in response.",
    )
    version: str = Field(
        default="v1",
        description="Model version to use: 'v1', 'v2', or 'ab' for random A/B routing.",
        pattern="^(v1|v2|ab)$",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return v.strip()


class BatchPredictRequest(BaseModel):
    """Batch review prediction request."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of review texts (1–32 items).",
    )
    return_probabilities: bool = Field(default=False)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        cleaned = []
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise ValueError(f"texts[{i}] must be a string")
            t = t.strip()
            if len(t) < 3:
                raise ValueError(f"texts[{i}] must be at least 3 characters")
            if len(t) > 2000:
                raise ValueError(f"texts[{i}] must not exceed 2000 characters")
            cleaned.append(t)
        return cleaned
```

**`app/schemas/response.py`**:

```python
"""Pydantic response schemas."""
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """Single prediction response."""

    sentiment: str = Field(..., description="Predicted sentiment label.")
    label_id: int = Field(..., description="Numeric label ID.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Max class probability.")
    probabilities: dict[str, float] | None = Field(
        default=None, description="Per-class probabilities (only if requested)."
    )
    processing_time_ms: float = Field(..., description="Total inference time in ms.")
    model_version: str = Field(..., description="Model version used.")

    model_config = {"json_schema_extra": {
        "example": {
            "sentiment": "positive",
            "label_id": 1,
            "confidence": 0.9847,
            "probabilities": {"negative": 0.0153, "positive": 0.9847},
            "processing_time_ms": 42.3,
            "model_version": "distilbert-sentiment-v1",
        }
    }}


class BatchResultItem(BaseModel):
    """Single item in a batch prediction response."""
    index: int
    text_preview: str
    sentiment: str
    confidence: float
    probabilities: dict[str, float] | None = None


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    results: list[BatchResultItem]
    total: int
    processing_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class ReadinessResponse(BaseModel):
    status: str
    model: str | None = None
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    base_model: str
    dataset: str
    labels: list[str]
    test_accuracy: float | None = None
    test_f1_macro: float | None = None
    training_date: str | None = None
    wandb_run_url: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    timestamp: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
```

---

### 3.4 Inference Service

**`app/services/inference.py`**:

```python
"""Core inference logic — tokenisation and model forward pass."""
import re
import time
import logging

import torch
import torch.nn.functional as F

from app.core.model import ModelBundle

logger = logging.getLogger(__name__)

MAX_LENGTH = 128


def clean_text(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_single(
    text: str,
    bundle: ModelBundle,
    return_probabilities: bool = False,
) -> dict:
    """
    Run inference on a single text string.

    Args:
        text: Raw review text (will be cleaned internally).
        bundle: Loaded ModelBundle.
        return_probabilities: Whether to return per-class probabilities.

    Returns:
        dict with sentiment, label_id, confidence, probabilities, processing_time_ms.
    """
    start = time.perf_counter()

    cleaned = clean_text(text)

    inputs = bundle.tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(bundle.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bundle.model(**inputs)

    logits = outputs.logits  # shape: (1, num_labels)
    probs = F.softmax(logits, dim=-1).squeeze(0)  # shape: (num_labels,)

    label_id = int(probs.argmax().item())
    confidence = float(probs[label_id].item())
    sentiment = bundle.labels[label_id]

    elapsed_ms = (time.perf_counter() - start) * 1000

    result = {
        "sentiment": sentiment,
        "label_id": label_id,
        "confidence": round(confidence, 6),
        "probabilities": None,
        "processing_time_ms": round(elapsed_ms, 2),
        "model_version": bundle.model_name,
    }

    if return_probabilities:
        result["probabilities"] = {
            bundle.labels[i]: round(float(probs[i].item()), 6)
            for i in range(len(bundle.labels))
        }

    return result


def predict_batch(
    texts: list[str],
    bundle: ModelBundle,
    return_probabilities: bool = False,
    batch_size: int = 16,
) -> list[dict]:
    """
    Run inference on a list of texts using micro-batching.

    Args:
        texts: List of raw review strings.
        bundle: Loaded ModelBundle.
        return_probabilities: Include per-class probs in output.
        batch_size: Internal micro-batch size for GPU efficiency.

    Returns:
        List of prediction dicts (same order as input).
    """
    cleaned_texts = [clean_text(t) for t in texts]
    all_results = []

    for i in range(0, len(cleaned_texts), batch_size):
        micro_batch = cleaned_texts[i : i + batch_size]

        inputs = bundle.tokenizer(
            micro_batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(bundle.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = bundle.model(**inputs)

        probs_batch = F.softmax(outputs.logits, dim=-1)  # (batch, num_labels)

        for j, probs in enumerate(probs_batch):
            label_id = int(probs.argmax().item())
            confidence = float(probs[label_id].item())
            sentiment = bundle.labels[label_id]

            item = {
                "sentiment": sentiment,
                "label_id": label_id,
                "confidence": round(confidence, 6),
                "probabilities": None,
            }
            if return_probabilities:
                item["probabilities"] = {
                    bundle.labels[k]: round(float(probs[k].item()), 6)
                    for k in range(len(bundle.labels))
                }
            all_results.append(item)

    return all_results
```

---

### 3.5 Main Application Factory

**`app/main.py`**:

```python
"""FastAPI application factory with lifespan management."""
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.model import load_model
from app.api.v1.routes import predict, batch, health

# ── Structured Logging Setup ──────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: load model on startup, cleanup on shutdown.
    Preferred over deprecated @app.on_event.
    """
    settings = get_settings()
    logger.info("startup.begin", model_path=settings.model_path)

    try:
        app.state.bundle_v1 = load_model(settings.model_path, version_tag="v1")
        app.state.model_loaded = True
        logger.info("startup.model_loaded", version="v1")
    except RuntimeError as e:
        logger.error("startup.model_load_failed", error=str(e))
        app.state.bundle_v1 = None
        app.state.model_loaded = False

    # Optional second model for A/B testing
    app.state.bundle_v2 = None
    if settings.model_path_v2:
        try:
            app.state.bundle_v2 = load_model(settings.model_path_v2, version_tag="v2")
            logger.info("startup.model_loaded", version="v2")
        except RuntimeError as e:
            logger.warning("startup.v2_load_failed", error=str(e))

    yield  # Application runs here

    logger.info("shutdown.begin")
    # Cleanup (free GPU memory if needed)
    if app.state.bundle_v1:
        del app.state.bundle_v1
    if app.state.bundle_v2:
        del app.state.bundle_v2
    logger.info("shutdown.complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Sentiment Analysis API",
        description=(
            "Fine-tuned DistilBERT model for product review sentiment classification. "
            "Trained on Amazon Reviews dataset with 92%+ accuracy."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID Middleware ─────────────────────────────────────────
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # ── Request Logging Middleware ────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        import time
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(elapsed_ms, 2),
            request_id=getattr(request.state, "request_id", "unknown"),
        )
        return response

    # ── Global Exception Handler ──────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error("unhandled_exception", error=str(exc), request_id=request_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred.",
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            },
        )

    # ── Routers ───────────────────────────────────────────────────────
    app.include_router(health.router, tags=["Health"])
    app.include_router(predict.router, prefix="/api/v1", tags=["Predictions"])
    app.include_router(batch.router, prefix="/api/v1", tags=["Predictions"])

    return app


app = create_app()
```

---

### 3.6 Smoke Test — Does It Start?

```bash
# Start the server
uvicorn app.main:app --reload --port 8000

# In another terminal
curl http://localhost:8000/health
# Expected: {"status":"ok","timestamp":"..."}

curl http://localhost:8000/ready
# Expected: {"status":"ready","model_loaded":true,...}

curl http://localhost:8000/docs
# Open in browser — should show Swagger UI
```

---

### 3.7 Phase 3 Acceptance Check

- [ ] `uvicorn app.main:app` starts without errors
- [ ] `/health` returns 200
- [ ] `/ready` returns `model_loaded: true` (or 503 if model not yet downloaded)
- [ ] `/docs` renders Swagger UI
- [ ] Structured JSON logs appear in terminal for each request
- [ ] `X-Request-ID` header present on all responses

---

## Phase 4 — API Endpoints, Middleware & Docs

**PRD Sections:** 7.3–7.6, 11.3–11.4  


---

### 4.1 Health Routes

**`app/api/v1/routes/health.py`**:

```python
"""Health and readiness check endpoints."""
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas.response import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Check",
    description="Confirms the service process is running. Use for load balancer liveness probes.",
)
async def health():
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/ready",
    summary="Readiness Check",
    description="Confirms the model is loaded and ready to serve predictions. Returns 503 if model is not loaded.",
)
async def ready(request: Request):
    loaded = getattr(request.app.state, "model_loaded", False)
    bundle = getattr(request.app.state, "bundle_v1", None)

    if loaded and bundle:
        return ReadinessResponse(
            status="ready",
            model=bundle.model_name,
            model_loaded=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    else:
        return JSONResponse(
            status_code=503,
            content=ReadinessResponse(
                status="not_ready",
                model=None,
                model_loaded=False,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
        )
```

---

### 4.2 Predict Route with Rate Limiting

**`app/api/v1/routes/predict.py`**:

```python
"""Single prediction and model info endpoints."""
import random
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.schemas.request import PredictRequest
from app.schemas.response import ModelInfoResponse, PredictResponse, ErrorResponse
from app.services.inference import predict_single

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


def get_bundle(request: Request, version: str = "v1"):
    """Resolve the correct ModelBundle based on version param."""
    if version == "ab":
        version = random.choice(["v1", "v2"])

    if version == "v2":
        bundle = getattr(request.app.state, "bundle_v2", None)
        if bundle is None:
            raise HTTPException(status_code=404, detail="Model v2 not loaded.")
        return bundle

    bundle = getattr(request.app.state, "bundle_v1", None)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "MODEL_NOT_READY",
                    "message": "Model is not loaded yet.",
                    "request_id": getattr(request.state, "request_id", "unknown"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            }
        )
    return bundle


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        422: {"description": "Validation Error", "model": ErrorResponse},
        429: {"description": "Rate Limit Exceeded", "model": ErrorResponse},
        503: {"description": "Model Not Ready", "model": ErrorResponse},
    },
    summary="Predict Sentiment",
    description=(
        "Classify a single product review as **positive** or **negative**. "
        "Returns confidence score and optional per-class probabilities."
    ),
)
@limiter.limit(f"{get_settings().rate_limit_per_minute}/minute")
async def predict(
    request: Request,
    body: PredictRequest,
):
    bundle = get_bundle(request, body.version)
    result = predict_single(
        text=body.text,
        bundle=bundle,
        return_probabilities=body.return_probabilities,
    )
    return PredictResponse(**result)


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Returns metadata about the currently loaded model.",
)
async def model_info(request: Request):
    bundle = getattr(request.app.state, "bundle_v1", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Try to read model card for metrics
    import json
    from pathlib import Path

    metadata = {}
    card_path = Path(bundle.model_path) / "model_card.md"
    args_path = Path(bundle.model_path) / "training_args.json"

    return ModelInfoResponse(
        model_name=bundle.model_name,
        base_model="distilbert-base-uncased",
        dataset="amazon_polarity (HuggingFace)",
        labels=list(bundle.labels.values()),
        test_accuracy=None,   # Populate from model_card.md if needed
        test_f1_macro=None,
        training_date=None,
        wandb_run_url=None,
    )
```

---

### 4.3 Batch Route

**`app/api/v1/routes/batch.py`**:

```python
"""Batch prediction endpoint."""
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.schemas.request import BatchPredictRequest
from app.schemas.response import BatchPredictResponse, BatchResultItem
from app.services.inference import predict_batch

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Batch Predict Sentiment",
    description=(
        "Classify up to **32 reviews** in a single request. "
        "Uses micro-batching internally for GPU efficiency."
    ),
)
@limiter.limit(f"{get_settings().rate_limit_per_minute}/minute")
async def batch_predict(
    request: Request,
    body: BatchPredictRequest,
):
    bundle = getattr(request.app.state, "bundle_v1", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    start = time.perf_counter()

    raw_results = predict_batch(
        texts=body.texts,
        bundle=bundle,
        return_probabilities=body.return_probabilities,
    )

    results = [
        BatchResultItem(
            index=i,
            text_preview=(body.texts[i][:50] + "...") if len(body.texts[i]) > 50 else body.texts[i],
            sentiment=r["sentiment"],
            confidence=r["confidence"],
            probabilities=r.get("probabilities"),
        )
        for i, r in enumerate(raw_results)
    ]

    elapsed_ms = (time.perf_counter() - start) * 1000

    return BatchPredictResponse(
        results=results,
        total=len(results),
        processing_time_ms=round(elapsed_ms, 2),
        model_version=bundle.model_name,
    )
```

---

### 4.4 Wire Up Rate Limiter in main.py

Add to `app/main.py` after imports:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add to create_app() before routers:
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

---

### 4.5 Manual API Testing

```bash
# Start server
uvicorn app.main:app --reload

# Test single predict
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!", "return_probabilities": true}'

# Test batch
curl -X POST http://localhost:8000/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible quality, broke in a day."], "return_probabilities": false}'

# Test model info
curl http://localhost:8000/api/v1/model/info

# Test validation error
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "hi"}'
# Expected: 422 with validation detail

# Test rate limit (run 65 times quickly)
for i in {1..65}; do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -X POST http://localhost:8000/api/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "test review text here"}';
done
# Last few should return 429
```

---

### 4.6 Phase 4 Acceptance Check

- [ ] All 5 endpoints return correct responses
- [ ] 422 on text < 3 chars
- [ ] 429 after rate limit breach
- [ ] 503 if model bundle missing from state
- [ ] Swagger at `/docs` shows all endpoints with examples
- [ ] Batch returns results in same order as input

---

## Phase 5 — Testing Suite

**PRD Sections:** 8.3, 12.5  


---

### 5.1 Test Fixtures with Mocked Model

**`tests/conftest.py`**:

```python
"""Shared test fixtures with mocked model — no GPU required."""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import torch
from app.main import create_app
from app.core.model import ModelBundle


def make_mock_bundle(version="v1"):
    """Create a minimal ModelBundle mock that returns deterministic predictions."""
    bundle = MagicMock(spec=ModelBundle)
    bundle.model_name = f"distilbert-sentiment-{version}"
    bundle.labels = {0: "negative", 1: "positive"}
    bundle.device = torch.device("cpu")

    # Mock tokenizer: returns input_ids and attention_mask tensors
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros(1, 128, dtype=torch.long),
        "attention_mask": torch.ones(1, 128, dtype=torch.long),
    }
    bundle.tokenizer = mock_tokenizer

    # Mock model: returns logits that favor "positive"
    mock_model_output = MagicMock()
    mock_model_output.logits = torch.tensor([[0.1, 2.5]])  # positive wins
    bundle.model = MagicMock(return_value=mock_model_output)

    return bundle


@pytest.fixture
def mock_bundle():
    return make_mock_bundle("v1")


@pytest.fixture
def client(mock_bundle):
    """TestClient with mocked model loaded into app state."""
    app = create_app()
    app.state.bundle_v1 = mock_bundle
    app.state.bundle_v2 = None
    app.state.model_loaded = True
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_no_model():
    """TestClient without model loaded — tests 503 responses."""
    app = create_app()
    app.state.bundle_v1 = None
    app.state.bundle_v2 = None
    app.state.model_loaded = False
    with TestClient(app) as c:
        yield c
```

---

### 5.2 Health Tests

**`tests/test_health.py`**:

```python
"""Tests for health and readiness endpoints."""
import pytest


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_health_has_request_id_header(client):
    response = client.get("/health")
    assert "x-request-id" in response.headers


def test_ready_returns_200_when_model_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True


def test_ready_returns_503_when_model_not_loaded(client_no_model):
    response = client_no_model.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["model_loaded"] is False
```

---

### 5.3 Predict Tests

**`tests/test_predict.py`**:

```python
"""Tests for single prediction endpoint."""
import pytest


VALID_REVIEW = "This is a great product that I really enjoyed using."


def test_predict_returns_200_for_valid_input(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": VALID_REVIEW},
    )
    assert response.status_code == 200


def test_predict_response_schema(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": VALID_REVIEW},
    )
    data = response.json()
    assert "sentiment" in data
    assert "label_id" in data
    assert "confidence" in data
    assert "processing_time_ms" in data
    assert "model_version" in data
    assert data["sentiment"] in ("positive", "negative")
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_with_probabilities(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": VALID_REVIEW, "return_probabilities": True},
    )
    data = response.json()
    assert data["probabilities"] is not None
    assert "positive" in data["probabilities"]
    assert "negative" in data["probabilities"]
    # Probabilities should sum to ~1.0
    total = sum(data["probabilities"].values())
    assert abs(total - 1.0) < 0.01


def test_predict_without_probabilities_by_default(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": VALID_REVIEW},
    )
    assert response.json()["probabilities"] is None


def test_predict_422_for_text_too_short(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": "hi"},
    )
    assert response.status_code == 422


def test_predict_422_for_empty_text(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": "   "},
    )
    assert response.status_code == 422


def test_predict_422_for_text_too_long(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": "x" * 2001},
    )
    assert response.status_code == 422


def test_predict_422_for_missing_text(client):
    response = client.post("/api/v1/predict", json={})
    assert response.status_code == 422


def test_predict_503_when_model_not_loaded(client_no_model):
    response = client_no_model.post(
        "/api/v1/predict",
        json={"text": VALID_REVIEW},
    )
    assert response.status_code == 503


def test_predict_html_stripped_from_input(client):
    """HTML in text should not crash inference."""
    response = client.post(
        "/api/v1/predict",
        json={"text": "<b>Amazing product!</b> Really loved it."},
    )
    assert response.status_code == 200


def test_predict_response_has_model_version(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": VALID_REVIEW},
    )
    assert "distilbert-sentiment" in response.json()["model_version"]
```

---

### 5.4 Batch Tests

**`tests/test_batch.py`**:

```python
"""Tests for batch prediction endpoint."""
import pytest


REVIEWS = [
    "Excellent product, very happy with my purchase.",
    "Complete garbage. Do not buy this.",
    "It is okay, nothing special.",
]


def test_batch_predict_returns_200(client):
    response = client.post("/api/v1/batch", json={"texts": REVIEWS})
    assert response.status_code == 200


def test_batch_predict_response_length_matches_input(client):
    response = client.post("/api/v1/batch", json={"texts": REVIEWS})
    data = response.json()
    assert data["total"] == len(REVIEWS)
    assert len(data["results"]) == len(REVIEWS)


def test_batch_predict_index_order_preserved(client):
    response = client.post("/api/v1/batch", json={"texts": REVIEWS})
    results = response.json()["results"]
    for i, r in enumerate(results):
        assert r["index"] == i


def test_batch_422_empty_list(client):
    response = client.post("/api/v1/batch", json={"texts": []})
    assert response.status_code == 422


def test_batch_422_more_than_32_items(client):
    texts = ["Some review text here."] * 33
    response = client.post("/api/v1/batch", json={"texts": texts})
    assert response.status_code == 422


def test_batch_422_item_too_short(client):
    response = client.post("/api/v1/batch", json={"texts": ["ok", "great product"]})
    assert response.status_code == 422


def test_batch_has_processing_time(client):
    response = client.post("/api/v1/batch", json={"texts": REVIEWS})
    assert "processing_time_ms" in response.json()
    assert response.json()["processing_time_ms"] > 0


def test_batch_503_no_model(client_no_model):
    response = client_no_model.post("/api/v1/batch", json={"texts": REVIEWS})
    assert response.status_code == 503
```

---

### 5.5 Run Tests

```bash
# Run full test suite with coverage
pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
# xdg-open htmlcov/index.html  # Linux

# Quick run (no coverage)
pytest tests/ -v
```

Expected output: ≥ 80% coverage, all tests green.

---

### 5.6 Phase 5 Acceptance Check

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Coverage ≥ 80%
- [ ] No real model download during tests (mock-only)
- [ ] Tests run in < 15 seconds

---

## Phase 6 — Dockerisation

**PRD Sections:** 8.5, 8.4 Job 3  


---

### 6.1 Multi-Stage Dockerfile

**`docker/Dockerfile`**:

```dockerfile
# ════════════════════════════════════════════════════════
# Stage 1: Builder — install dependencies
# ════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /install

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ════════════════════════════════════════════════════════
# Stage 2: Runtime — minimal final image
# ════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/lib /usr/local/lib
COPY --from=builder /install/bin /usr/local/bin

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app

# Copy application source
COPY app/ ./app/
COPY scripts/ ./scripts/

# Model dir — populated at runtime via download_model.sh or volume mount
VOLUME ["/app/models"]

USER appuser

EXPOSE 8000

ENV PORT=8000 \
    MODEL_PATH=/app/models/distilbert-sentiment \
    LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

---

### 6.2 .dockerignore

**`.dockerignore`**:

```
.git
.venv
.env
notebooks/
tests/
*.md
*.zip
*.tar.gz
__pycache__/
*.pyc
*.pyo
.pytest_cache/
htmlcov/
.coverage
wandb/
reports/
models/distilbert-sentiment/
docker-compose*.yml
.github/
sweeps/
monitoring/
```

---

### 6.3 Docker Compose for Local Development

**`docker/docker-compose.yml`**:

```yaml
version: "3.9"

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: sentiment-api:latest
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/distilbert-sentiment
      - LOG_LEVEL=INFO
      - ALLOWED_ORIGINS=*
      - RATE_LIMIT_PER_MINUTE=60
    volumes:
      # Mount local model directory so you don't rebuild image on model changes
      - ../models:/app/models:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

---

### 6.4 Model Download Script

**`scripts/download_model.sh`**:

```bash
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

model_id = os.environ.get('HF_MODEL_ID', 'your-username/distilbert-sentiment-amazon')
save_path = os.environ.get('MODEL_PATH', './models/distilbert-sentiment')

print(f'Downloading {model_id} to {save_path}...')
tokenizer = DistilBertTokenizer.from_pretrained(model_id)
model = DistilBertForSequenceClassification.from_pretrained(model_id)
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
```

```bash
chmod +x scripts/download_model.sh
```

---

### 6.5 Build and Test Docker Image

```bash
# Build from project root
docker build -f docker/Dockerfile -t sentiment-api:latest .

# Check image size
docker images sentiment-api:latest

# Run with local model mounted
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_PATH=/app/models/distilbert-sentiment \
  sentiment-api:latest

# Smoke test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The product quality is outstanding!"}'
```

---

### 6.6 Phase 6 Acceptance Check

- [ ] `docker build` succeeds without errors
- [ ] Image size < 1.5 GB (model not baked in)
- [ ] Container starts and `/health` returns 200
- [ ] Non-root user running inside container (`docker exec <id> whoami` → `appuser`)
- [ ] `docker-compose up` from `docker/` directory starts the service

---

## Phase 7 — CI/CD with GitHub Actions

**PRD Sections:** 8.3–8.4  


---

### 7.1 CI Pipeline

**`.github/workflows/ci.yml`**:

```yaml
name: CI — Lint, Test, Docker Smoke

on:
  push:
    branches: ["**"]
  pull_request:
    branches: [main]

jobs:
  # ── Job 1: Lint ────────────────────────────────────────────────────
  lint:
    name: Lint (ruff + black + isort)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: lint-${{ hashFiles('requirements-dev.txt') }}

      - name: Install linters
        run: pip install ruff==0.4.4 black==24.4.2 isort==5.13.2

      - name: Run ruff
        run: ruff check app/ tests/

      - name: Run black
        run: black --check app/ tests/

      - name: Run isort
        run: isort --check-only app/ tests/

  # ── Job 2: Test ────────────────────────────────────────────────────
  test:
    name: Test (pytest + coverage)
    runs-on: ubuntu-latest
    needs: lint

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: test-${{ hashFiles('requirements.txt', 'requirements-dev.txt') }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests with coverage
        run: |
          pytest tests/ -v \
            --cov=app \
            --cov-report=xml \
            --cov-report=term-missing \
            --cov-fail-under=80

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: coverage.xml
          fail_ci_if_error: false

  # ── Job 3: Docker Build Smoke Test ────────────────────────────────
  docker-build:
    name: Docker Build + Smoke Test
    runs-on: ubuntu-latest
    needs: test

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile
          push: false
          tags: sentiment-api:ci-test
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Docker smoke test
        run: |
          docker run -d \
            --name smoke-test \
            -p 8000:8000 \
            -e MODEL_PATH=/app/models/distilbert-sentiment \
            sentiment-api:ci-test

          # Wait for container to start (up to 30s)
          for i in {1..15}; do
            if curl -sf http://localhost:8000/health; then
              echo "✅ Health check passed"
              break
            fi
            echo "Waiting... ($i/15)"
            sleep 2
          done

          # Assert health returns 200
          STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
          if [ "$STATUS" != "200" ]; then
            echo "❌ Health check returned $STATUS"
            docker logs smoke-test
            exit 1
          fi

          docker stop smoke-test
```

---

### 7.2 CD Pipeline

**`.github/workflows/deploy.yml`**:

```yaml
name: CD — Build, Push & Deploy

on:
  push:
    branches: [main]

jobs:
  # ── Job 1: Build and push to GHCR ─────────────────────────────────
  build-and-push:
    name: Build & Push Docker Image
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=raw,value=latest
            type=sha,prefix=sha-

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ── Job 2: Deploy to Railway ───────────────────────────────────────
  deploy-railway:
    name: Deploy to Railway
    runs-on: ubuntu-latest
    needs: build-and-push
    environment: production

    steps:
      - uses: actions/checkout@v4

      - name: Install Railway CLI
        run: npm install -g @railway/cli

      - name: Deploy to Railway
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
        run: railway up --service sentiment-api --detach

      - name: Wait for deployment health
        run: |
          DEPLOY_URL="${{ secrets.RAILWAY_PUBLIC_URL }}"
          echo "Waiting for $DEPLOY_URL to be healthy..."
          for i in {1..20}; do
            STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$DEPLOY_URL/health" || echo "000")
            if [ "$STATUS" = "200" ]; then
              echo "✅ Deployment healthy at $DEPLOY_URL"
              exit 0
            fi
            echo "Status: $STATUS — retrying ($i/20)..."
            sleep 15
          done
          echo "❌ Deployment health check timed out"
          exit 1

      - name: Post deployment status comment
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `✅ Deployed to production: ${{ secrets.RAILWAY_PUBLIC_URL }}`
            })
```

---

### 7.3 GitHub Repository Secrets Setup

In your GitHub repo → Settings → Secrets and variables → Actions:

| Secret Name | Value | Purpose |
|-------------|-------|---------|
| `RAILWAY_TOKEN` | Railway CLI token | CD deployment |
| `RAILWAY_PUBLIC_URL` | `https://your-app.railway.app` | Health check URL |
| `CODECOV_TOKEN` | Codecov.io token | Coverage upload |
| `WANDB_API_KEY` | W&B API key | Optional: for drift monitoring |

Never put these values in source code.

---

### 7.4 Phase 7 Acceptance Check

- [ ] Push to any branch triggers CI (lint → test → docker-build)
- [ ] All CI jobs pass on a clean commit
- [ ] Coverage report uploaded to Codecov
- [ ] Merge to `main` triggers CD
- [ ] GHCR shows the pushed image
- [ ] No secrets appear anywhere in source code

---

## Phase 8 — Deployment to Railway

**PRD Sections:** 8.4, 11.1  


---

### 8.1 Railway Project Setup

1. Go to [railway.app](https://railway.app) → New Project
2. Choose "Deploy from GitHub repo" → select `sentiment-analysis-api`
3. Railway auto-detects Dockerfile at `docker/Dockerfile`

---

### 8.2 Railway Environment Variables

In Railway → your service → Variables, add:

```
MODEL_PATH=/app/models/distilbert-sentiment
HF_MODEL_ID=your-hf-username/distilbert-sentiment-amazon
LOG_LEVEL=INFO
ALLOWED_ORIGINS=*
RATE_LIMIT_PER_MINUTE=60
PORT=8000
```

---

### 8.3 Model at Runtime (Railway Build Command)

Since the model is not in the Docker image, we need Railway to download it at startup.

Add a `railway.toml` to the project root:

```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "docker/Dockerfile"

[deploy]
startCommand = "bash scripts/download_model.sh hf && uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2"
healthcheckPath = "/health"
healthcheckTimeout = 120
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

This downloads the model from HuggingFace Hub every cold start (acceptable for a portfolio project; for production use Railway persistent volumes or a model registry).

---

### 8.4 Verify Deployment

```bash
# Get your Railway URL from the dashboard
export RAILWAY_URL="https://your-app.railway.app"

# Health check
curl $RAILWAY_URL/health

# Live prediction
curl -X POST $RAILWAY_URL/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely incredible, best thing I have bought.", "return_probabilities": true}'

# Docs
open $RAILWAY_URL/docs
```

---

### 8.5 Set Up UptimeRobot (Free Uptime Monitoring)

1. Go to [uptimerobot.com](https://uptimerobot.com) → Add New Monitor
2. Type: HTTP(s)
3. URL: `https://your-app.railway.app/health`
4. Monitoring interval: 5 minutes
5. Alert contacts: your email

---

### 8.6 Phase 8 Acceptance Check

- [ ] Railway deployment shows "Active" / green status
- [ ] `curl $RAILWAY_URL/health` returns 200
- [ ] `curl $RAILWAY_URL/ready` returns `model_loaded: true`
- [ ] Swagger UI accessible at `$RAILWAY_URL/docs`
- [ ] Live prediction returns correct response
- [ ] UptimeRobot shows green

---

## Phase 9 — Stand-Out Features

**PRD Sections:** 13.1–13.7  
 
**Priority order:** W&B Sweeps → Prometheus Metrics → API Key Auth → Drift Detection → Async Batch → Star Rating → A/B Testing

---

### 9.1 W&B Hyperparameter Sweep (13.1)

Create **`sweeps/sweep_config.yaml`**:

```yaml
program: scripts/train_sweep.py
method: bayes
metric:
  name: val/f1_macro
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 5e-5
  per_device_train_batch_size:
    values: [8, 16]
  warmup_ratio:
    values: [0.05, 0.1, 0.2]
  weight_decay:
    values: [0.0, 0.01, 0.05]
early_terminate:
  type: hyperband
  min_iter: 1
```

Create **`scripts/train_sweep.py`** — a single training run for the sweep agent:

```python
"""Single training run called by W&B sweep agent."""
import wandb
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
# ... (same training setup as notebook but driven by wandb.config)

def train():
    with wandb.init() as run:
        config = wandb.config
        set_seed(42)
        # Load pre-tokenized dataset cached from notebook
        # ... train with config.learning_rate, config.per_device_train_batch_size, etc.

if __name__ == "__main__":
    train()
```

Run the sweep:

```bash
# Initialize sweep (returns sweep ID)
wandb sweep sweeps/sweep_config.yaml --project sentiment-analysis-distilbert

# Launch agent (run 10 trials)
wandb agent <sweep-id> --count 10
```

---

### 9.2 Prometheus Metrics (13.4)

Install:

```bash
pip install prometheus-fastapi-instrumentator==7.0.0
```

Add to **`app/main.py`** in `create_app()`:

```python
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# Custom ML metrics
sentiment_counter = Counter(
    "sentiment_prediction_total",
    "Total predictions by sentiment label",
    ["label", "model_version"],
)
confidence_histogram = Histogram(
    "sentiment_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
inference_duration = Histogram(
    "inference_duration_seconds",
    "Model inference duration in seconds",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
)

# Auto-instrument FastAPI
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

Update `predict_single()` in `app/services/inference.py` to record metrics:

```python
# After computing result:
from app.main import sentiment_counter, confidence_histogram, inference_duration

sentiment_counter.labels(
    label=result["sentiment"],
    model_version=bundle.model_name
).inc()
confidence_histogram.observe(result["confidence"])
inference_duration.observe(elapsed_ms / 1000)
```

**Prometheus + Grafana compose file** — **`monitoring/docker-compose.monitoring.yml`**:

```yaml
version: "3.9"
services:
  prometheus:
    image: prom/prometheus:v2.52.0
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:10.4.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./grafana/dashboards:/var/lib/grafana/dashboards
```

**`monitoring/prometheus/prometheus.yml`**:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: sentiment-api
    static_configs:
      - targets: ["host.docker.internal:8000"]
    metrics_path: /metrics
```

Run locally:

```bash
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

---

### 9.3 API Key Authentication with Tiered Rate Limits (13.3)

Add to **`app/core/middleware.py`**:

```python
"""API key authentication and tiered rate limiting."""
import hashlib
from fastapi import Request, HTTPException
from app.core.config import get_settings


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def verify_api_key(request: Request) -> dict:
    """
    Check X-API-Key header. Returns tier info.
    Raises 401 if key is invalid (when auth is enabled).
    """
    settings = get_settings()
    if not settings.api_keys_set:
        # Auth disabled — anonymous access
        return {"tier": "anonymous", "limit": settings.rate_limit_per_minute}

    raw_key = request.headers.get("X-API-Key")
    if not raw_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required.")

    hashed = hash_api_key(raw_key)
    if hashed not in settings.api_keys_set:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    return {"tier": "authenticated", "limit": settings.rate_limit_per_minute * 5}
```

Add validation endpoint to a new file **`app/api/v1/routes/keys.py`**:

```python
from fastapi import APIRouter, Request, Header
from app.core.middleware import verify_api_key

router = APIRouter()

@router.get("/keys/validate", summary="Validate API Key")
async def validate_key(request: Request, x_api_key: str = Header(None)):
    key_info = await verify_api_key(request)
    return {"valid": True, "tier": key_info["tier"]}
```

To generate a key for testing:

```python
import hashlib, secrets
raw = secrets.token_hex(32)
hashed = hashlib.sha256(raw.encode()).hexdigest()
print(f"Your key (share this): {raw}")
print(f"Store this in API_KEYS env var: {hashed}")
```

---

### 9.4 Drift Detection with Evidently AI (13.5)

Install:

```bash
pip install evidently==0.4.33 apscheduler==3.10.4 aiosqlite==0.20.0
```

Create **`app/services/drift.py`**:

```python
"""Input drift detection using Evidently AI."""
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

logger = logging.getLogger(__name__)
DB_PATH = "./data/request_log.db"
REPORTS_DIR = Path("./reports")
MAX_LOG_SIZE = 1000  # Keep last N requests


def init_db():
    """Initialize SQLite database for request logging."""
    Path(DB_PATH).parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS request_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            text_length INTEGER,
            word_count INTEGER,
            predicted_label TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()


def log_request(text: str, sentiment: str, confidence: float):
    """Log a prediction request to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO request_log (timestamp, text_length, word_count, predicted_label, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), len(text), len(text.split()), sentiment, confidence))
    # Trim to MAX_LOG_SIZE
    conn.execute(f"""
        DELETE FROM request_log WHERE id NOT IN (
            SELECT id FROM request_log ORDER BY id DESC LIMIT {MAX_LOG_SIZE}
        )
    """)
    conn.commit()
    conn.close()


def run_drift_check(reference_stats_path: str = "./data/training_stats.json"):
    """
    Compare recent API traffic stats against training distribution.
    Logs warning if drift detected.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        current_df = pd.read_sql("SELECT * FROM request_log ORDER BY id DESC LIMIT 500", conn)
        conn.close()

        if len(current_df) < 100:
            logger.info("drift.skip", reason="insufficient_data", count=len(current_df))
            return

        with open(reference_stats_path) as f:
            ref_data = json.load(f)

        ref_df = pd.DataFrame(ref_data)

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=current_df[["text_length", "word_count"]])

        REPORTS_DIR.mkdir(exist_ok=True)
        report_path = REPORTS_DIR / f"drift_{datetime.utcnow().strftime('%Y%m%d')}.html"
        report.save_html(str(report_path))

        drift_detected = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
        if drift_detected:
            logger.warning("drift.detected", report_path=str(report_path))
        else:
            logger.info("drift.none_detected", report_path=str(report_path))

    except Exception as e:
        logger.error("drift.check_failed", error=str(e))
```

Wire APScheduler into `app/main.py` lifespan:

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.services.drift import init_db, run_drift_check

# In lifespan, after model load:
init_db()
scheduler = AsyncIOScheduler()
scheduler.add_job(run_drift_check, "interval", hours=24)
scheduler.start()
app.state.scheduler = scheduler

# In yield/shutdown:
app.state.scheduler.shutdown()
```

---

### 9.5 Async Batch Processing (13.6)

Create **`app/api/v1/routes/jobs.py`**:

```python
"""Async batch job endpoints for large-scale prediction."""
import sqlite3
import uuid
import json
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.inference import predict_batch

router = APIRouter()
DB_PATH = "./data/jobs.db"


def init_jobs_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT,
            created_at TEXT,
            completed_at TEXT,
            total INTEGER,
            results TEXT
        )
    """)
    conn.commit()
    conn.close()


class AsyncBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=1000)


@router.post("/jobs/batch", summary="Submit Async Batch Job")
async def submit_batch_job(
    request: Request,
    body: AsyncBatchRequest,
    background_tasks: BackgroundTasks,
):
    job_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO jobs (id, status, created_at, total) VALUES (?, 'pending', ?, ?)",
        (job_id, datetime.now(timezone.utc).isoformat(), len(body.texts))
    )
    conn.commit()
    conn.close()

    bundle = getattr(request.app.state, "bundle_v1", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    async def process_job():
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE jobs SET status='processing' WHERE id=?", (job_id,))
        conn.commit()
        try:
            results = predict_batch(body.texts, bundle)
            conn.execute(
                "UPDATE jobs SET status='complete', results=?, completed_at=? WHERE id=?",
                (json.dumps(results), datetime.now(timezone.utc).isoformat(), job_id)
            )
        except Exception as e:
            conn.execute(
                "UPDATE jobs SET status='failed', results=? WHERE id=?",
                (json.dumps({"error": str(e)}), job_id)
            )
        conn.commit()
        conn.close()

    background_tasks.add_task(process_job)
    return {"job_id": job_id, "status": "pending", "total": len(body.texts)}


@router.get("/jobs/{job_id}", summary="Get Job Status")
async def get_job_status(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT id, status, created_at, completed_at, total FROM jobs WHERE id=?",
        (job_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"job_id": row[0], "status": row[1], "created_at": row[2],
            "completed_at": row[3], "total": row[4]}


@router.get("/jobs/{job_id}/results", summary="Get Job Results")
async def get_job_results(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT status, results FROM jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")
    if row[0] != "complete":
        raise HTTPException(status_code=400, detail=f"Job status is '{row[0]}', not complete.")
    return {"job_id": job_id, "results": json.loads(row[1])}
```

---

### 9.6 Phase 9 Acceptance Check

Implement at least 3 stand-out features and verify:

- [ ] W&B Sweep: sweep runs and best config is identified
- [ ] `/metrics` endpoint returns Prometheus metrics including custom sentiment counters
- [ ] API key authentication works — unauthenticated gets 401 when `API_KEYS` is set
- [ ] Drift report generated in `./reports/` directory
- [ ] Async batch job returns `job_id`, status polling works, results retrievable

---

## Phase 10 — Polish, Load Testing & README

**PRD Sections:** 8.2, 12.1–12.3  


---

### 10.1 Load Testing with Locust

Create **`scripts/locustfile.py`**:

```python
"""Locust load test for Sentiment Analysis API."""
import random
from locust import HttpUser, task, between

SAMPLE_REVIEWS = [
    "This product is absolutely amazing! Best purchase I've ever made.",
    "Terrible quality. It broke within the first week of use.",
    "Decent product for the price. Nothing extraordinary but gets the job done.",
    "Would highly recommend to anyone looking for a reliable solution.",
    "The worst product I have ever bought. Complete waste of money.",
    "Pretty good overall. A few minor issues but nothing major.",
    "Five stars! Exceeded all my expectations in every way possible.",
    "Disappointing. The description was very misleading about what this product actually does.",
]


class SentimentAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)  # 0.1–0.5s between requests → ~10 RPS

    @task(8)
    def predict_single(self):
        text = random.choice(SAMPLE_REVIEWS)
        self.client.post(
            "/api/v1/predict",
            json={"text": text},
            name="/api/v1/predict",
        )

    @task(2)
    def predict_batch(self):
        texts = random.sample(SAMPLE_REVIEWS, k=random.randint(2, 5))
        self.client.post(
            "/api/v1/batch",
            json={"texts": texts},
            name="/api/v1/batch",
        )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")
```

Run the load test:

```bash
# Install locust
pip install locust==2.29.0

# Run against local server (start server first)
locust -f scripts/locustfile.py \
  --host http://localhost:8000 \
  --users 10 \
  --spawn-rate 2 \
  --run-time 60s \
  --headless \
  --html reports/load_test_report.html

# Open the report
open reports/load_test_report.html
```

Verify p50 < 100ms and p99 < 300ms for `/api/v1/predict`.

---

### 10.2 Final README.md

Replace the auto-generated README with a comprehensive project README:

```markdown
# Sentiment Analysis API

[![CI](https://github.com/<username>/sentiment-analysis-api/actions/workflows/ci.yml/badge.svg)](https://github.com/<username>/sentiment-analysis-api/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/<username>/sentiment-analysis-api/branch/main/graph/badge.svg)](https://codecov.io/gh/<username>/sentiment-analysis-api)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fine-tuned DistilBERT model for product review sentiment classification,
deployed as a production-grade REST API. End-to-end MLOps project demonstrating
the complete model lifecycle from data ingestion to live serving.

**Live API:** https://your-app.railway.app  
**API Docs:** https://your-app.railway.app/docs  
**W&B Dashboard:** https://wandb.ai/your-username/sentiment-analysis-distilbert  

---

## Architecture

[Include ASCII diagram from PRD Section 5]

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 92.3% |
| F1 Macro | 91.8% |
| ROC-AUC | 97.4% |
| p50 Latency | 72 ms |
| p99 Latency | 218 ms |

## Quickstart

### Local (Docker)
docker pull ghcr.io/<username>/sentiment-analysis-api:latest
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  ghcr.io/<username>/sentiment-analysis-api:latest

### Local (Python)
git clone https://github.com/<username>/sentiment-analysis-api
cd sentiment-analysis-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
MODEL_PATH=./models/distilbert-sentiment uvicorn app.main:app --reload

## API Usage

### Single Prediction
curl -X POST https://your-app.railway.app/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Best product I have ever bought!", "return_probabilities": true}'

### Batch Prediction
curl -X POST https://your-app.railway.app/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible quality, broke immediately."]}'

## Reproduce Training

1. Open notebooks/train_sentiment_distilbert.ipynb in Google Colab
2. Set WANDB_API_KEY in Colab secrets
3. Run all cells — training takes ~2.5 hours on T4 GPU
4. Download model from W&B artefact or HF Hub

## Stand-Out Features

- **W&B Hyperparameter Sweep** — Bayesian optimization across 10 runs
- **Prometheus Metrics** — Custom ML metrics at /metrics
- **API Key Auth** — Tiered rate limits (60 vs 300 req/min)
- **Drift Detection** — Daily Evidently AI reports
- **Async Batch Jobs** — Up to 1000 texts per job with polling

## CI/CD

Every PR runs: lint (ruff + black + isort) → tests (pytest + coverage) → Docker smoke test  
Every merge to main: build multi-platform image → push to GHCR → deploy to Railway

## License

MIT — see [LICENSE](LICENSE)
```

---

### 10.3 Final Commit Checklist

```bash
# Format everything
black app/ tests/ scripts/
isort app/ tests/ scripts/
ruff check app/ tests/ --fix

# Run full test suite one more time
pytest tests/ -v --cov=app

# Verify Docker build
docker build -f docker/Dockerfile -t sentiment-api:final .

# Final commit
git add -A
git commit -m "feat: complete project — all PRD requirements met"
git push
```

---

### 10.4 Phase 10 Acceptance Check

- [ ] Load test: p50 < 100ms, p99 < 300ms at 10 RPS
- [ ] Load test HTML report saved to `reports/`
- [ ] README has all 12 required sections (per PRD 8.2)
- [ ] All CI checks green on `main` branch
- [ ] Live Railway deployment healthy
- [ ] W&B run publicly viewable
- [ ] `model_card.md` committed to repo root

---

## Appendix A — Full File Tree

```
sentiment-analysis-api/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── routes/
│   │           ├── __init__.py
│   │           ├── health.py
│   │           ├── predict.py
│   │           ├── batch.py
│   │           ├── jobs.py          # stand-out: async batch
│   │           └── keys.py          # stand-out: api key auth
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── middleware.py
│   │   └── model.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request.py
│   │   └── response.py
│   └── services/
│       ├── __init__.py
│       ├── inference.py
│       └── drift.py                 # stand-out: drift detection
├── data/
│   └── prepare_dataset.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/
│   └── .gitkeep
├── monitoring/
│   ├── docker-compose.monitoring.yml
│   ├── grafana/
│   │   └── dashboards/
│   │       └── sentiment_api.json
│   └── prometheus/
│       └── prometheus.yml
├── notebooks/
│   └── train_sentiment_distilbert.ipynb
├── reports/
│   └── .gitkeep
├── scripts/
│   ├── download_model.sh
│   ├── locustfile.py
│   └── train_sweep.py
├── sweeps/
│   └── sweep_config.yaml
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_predict.py
│   └── test_batch.py
├── .dockerignore
├── .env.example
├── .gitignore
├── model_card.md
├── pyproject.toml
├── railway.toml
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── requirements-standout.txt
└── requirements-train.txt
```

---

## Appendix B — Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_PATH` | Yes | `./models/distilbert-sentiment` | Local model path or HF Hub ID |
| `MODEL_PATH_V2` | No | `` | Second model for A/B testing |
| `PORT` | No | `8000` | Server port |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |
| `ALLOWED_ORIGINS` | No | `*` | CORS allowed origins |
| `RATE_LIMIT_PER_MINUTE` | No | `60` | Requests/min per IP |
| `API_KEYS` | No | `` | Comma-separated SHA-256 key hashes |
| `WANDB_API_KEY` | Training | — | W&B authentication |
| `WANDB_PROJECT` | Training | `sentiment-analysis-distilbert` | W&B project |
| `WANDB_ENTITY` | Training | — | W&B username/team |
| `HF_TOKEN` | Optional | — | HF Hub push access |
| `HF_PUSH` | Optional | `false` | Push model to HF Hub |
| `HF_MODEL_ID` | Deploy | — | HF Hub model ID for download |

---

## Appendix C — Troubleshooting

### Training

**"CUDA out of memory"**  
Reduce `BATCH_SIZE` from 16 to 8. In Colab: Runtime → Change runtime type → T4 GPU.

**"wandb: ERROR Run failed"**  
Check `WANDB_API_KEY` is set. Run `wandb login` in the terminal.

**"Accuracy below 92% threshold"**  
Try: increase epochs to 4, reduce learning rate to 1e-5, increase train samples to 300k. Check class balance in splits.

### API

**"RuntimeError: No config.json found"**  
Model files not downloaded. Run `scripts/download_model.sh` first, or check `MODEL_PATH` env var.

**"Model not ready" (503 on /ready)**  
Model is still loading (can take 30–60s). Wait for logs to show "startup.model_loaded".

**"Rate limit exceeded" during testing**  
Set `RATE_LIMIT_PER_MINUTE=1000` in `.env` for local testing.

### CI/CD

**"black --check failed" in CI**  
Run `black app/ tests/` locally and commit the formatting changes.

**Docker build fails: "No module named X"**  
Library missing from `requirements.txt`. Add it, rebuild.

**Railway deployment: model download timeout**  
Increase `healthcheckTimeout` in `railway.toml` to 180. Consider caching model in a Railway volume.

---

*Dev Plan v1.0 — All sections map 1:1 to PRD v1.0 requirements. Update this document whenever the PRD changes.*
