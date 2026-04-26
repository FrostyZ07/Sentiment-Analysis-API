"""
DistilBERT Sentiment Analysis — Training Notebook
==================================================
Run in Google Colab with T4 GPU.

Instructions:
  1. Upload this file to Colab (or copy cells into a new notebook)
  2. Runtime → Change runtime type → T4 GPU
  3. Run all cells in order
  4. Download the model zip at the end

Each section marked with # %% is a separate Colab cell.
"""

# %% Cell 0: Install dependencies
import subprocess
subprocess.check_call([
    "pip", "install", "-q",
    "transformers", "accelerate", "datasets", "wandb",
])
print("✅ Done. Now restart runtime (Runtime → Restart) and skip this cell.")

# %% Cell 1: Imports and reproducibility
import os
import random
import re

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
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Constants ────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
TRAIN_SAMPLES = 200_000
VAL_SAMPLES = 25_000
TEST_SAMPLES = 25_000
LABEL_NAMES = ["negative", "positive"]

# %% Cell 2: W&B initialization
wandb.login()

run = wandb.init(
    project="sentiment-analysis-distilbert",
    name="eda-and-training",
    tags=["eda", "training", "full-run"],
    config={
        "seed": SEED,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "train_samples": TRAIN_SAMPLES,
        "val_samples": VAL_SAMPLES,
        "test_samples": TEST_SAMPLES,
        "dataset": "amazon_polarity",
    },
)
print(f"W&B Run URL: {run.get_url()}")

# %% Cell 3: Load amazon_polarity dataset
dataset = load_dataset("amazon_polarity")

print("Dataset structure:")
print(dataset)
print("\nFeatures:", dataset["train"].features)
print("\nTrain size:", len(dataset["train"]))
print("Test size:", len(dataset["test"]))

# %% Cell 4: Class distribution analysis
import matplotlib.pyplot as plt
import pandas as pd

# Compute class counts directly from dataset (no .to_pandas())
labels = dataset["train"]["label"]  # just the label column, lightweight
neg_count = labels.count(0)
pos_count = labels.count(1)

label_counts = pd.DataFrame({
    "label_name": ["negative", "positive"],
    "count": [neg_count, pos_count],
})

print("Class Distribution (Full Train Set):")
print(label_counts)
print(f"\nClass balance ratio: {min(neg_count, pos_count) / max(neg_count, pos_count):.4f}")

# Log to W&B
wandb.log({
    "class_distribution": wandb.plot.bar(
        wandb.Table(dataframe=label_counts),
        "label_name", "count",
        title="Class Distribution — Full Training Set",
    )
})

del labels  # free memory

# %% Cell 5: Text length distribution (using sample to save RAM)
EDA_SAMPLE_SIZE = 5000
eda_sample = dataset["train"].shuffle(seed=SEED).select(range(EDA_SAMPLE_SIZE))
sample_df = eda_sample.to_pandas()

sample_df["text_length"] = sample_df["content"].str.len()
sample_df["word_count"] = sample_df["content"].str.split().str.len()

print(f"Text length statistics (sample of {EDA_SAMPLE_SIZE}, characters):")
print(sample_df["text_length"].describe())

print(f"\nWord count statistics (sample of {EDA_SAMPLE_SIZE}):")
print(sample_df["word_count"].describe())

# Token length analysis
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
token_lengths = sample_df["content"].head(500).apply(
    lambda x: len(tokenizer.encode(x, add_special_tokens=True))
)
pct_over_128 = (token_lengths > 128).mean() * 100
pct_over_256 = (token_lengths > 256).mean() * 100
pct_over_512 = (token_lengths > 512).mean() * 100

print(f"\nToken length analysis (sample of 500):")
print(f"  > 128 tokens: {pct_over_128:.1f}%")
print(f"  > 256 tokens: {pct_over_256:.1f}%")
print(f"  > 512 tokens: {pct_over_512:.1f}%")

wandb.log({
    "token_length_distribution": wandb.plot.histogram(
        wandb.Table(data=[[l] for l in token_lengths], columns=["token_length"]),
        "token_length",
        title="Token Length Distribution (sample n=500)",
    ),
    "pct_reviews_over_128_tokens": pct_over_128,
    "pct_reviews_over_512_tokens": pct_over_512,
})

# %% Cell 6: Sample reviews per class
print("=" * 60)
print("NEGATIVE REVIEWS (label=0) — Sample:")
print("=" * 60)
neg_samples = sample_df[sample_df["label"] == 0]["content"].head(3)
for i, text in enumerate(neg_samples):
    print(f"\n[{i+1}] {text[:300]}...")

print("\n" + "=" * 60)
print("POSITIVE REVIEWS (label=1) — Sample:")
print("=" * 60)
pos_samples = sample_df[sample_df["label"] == 1]["content"].head(3)
for i, text in enumerate(pos_samples):
    print(f"\n[{i+1}] {text[:300]}...")

del sample_df, eda_sample  # free memory

# %% Cell 7: Preprocessing utilities


def clean_text(text: str) -> str:
    """Clean review text before tokenisation."""
    text = re.sub(r"<[^>]+>", " ", text)
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

# %% Cell 8: Build sampled, stratified splits
import datasets as hf_datasets
from datasets import DatasetDict


def prepare_splits(dataset, train_n, val_n, test_n, seed):
    """Sample stratified subsets from amazon_polarity."""

    def preprocess(example):
        example["content"] = clean_text(example["content"])
        return example

    dataset = dataset.map(preprocess, num_proc=2)
    dataset = dataset.filter(lambda x: is_valid_review(x["content"]))

    full_train = dataset["train"]
    neg_train = full_train.filter(lambda x: x["label"] == 0)
    pos_train = full_train.filter(lambda x: x["label"] == 1)

    neg_train_sample = neg_train.shuffle(seed=seed).select(range(train_n // 2))
    pos_train_sample = pos_train.shuffle(seed=seed).select(range(train_n // 2))
    train_split = hf_datasets.concatenate_datasets(
        [neg_train_sample, pos_train_sample]
    )
    train_split = train_split.shuffle(seed=seed)

    full_test = dataset["test"]
    neg_test = full_test.filter(lambda x: x["label"] == 0)
    pos_test = full_test.filter(lambda x: x["label"] == 1)

    neg_val = neg_test.shuffle(seed=seed).select(range(val_n // 2))
    pos_val = pos_test.shuffle(seed=seed).select(range(val_n // 2))
    val_split = hf_datasets.concatenate_datasets([neg_val, pos_val]).shuffle(
        seed=seed
    )

    neg_test_final = (
        neg_test.shuffle(seed=seed + 1).select(
            range(val_n, val_n + test_n // 2)
        )
    )
    pos_test_final = (
        pos_test.shuffle(seed=seed + 1).select(
            range(val_n, val_n + test_n // 2)
        )
    )
    test_split = hf_datasets.concatenate_datasets(
        [neg_test_final, pos_test_final]
    ).shuffle(seed=seed)

    return DatasetDict(
        {
            "train": train_split,
            "validation": val_split,
            "test": test_split,
        }
    )


splits = prepare_splits(dataset, TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES, SEED)
print("Split sizes:")
for name, ds in splits.items():
    neg = sum(1 for x in ds if x["label"] == 0)
    pos = sum(1 for x in ds if x["label"] == 1)
    print(f"  {name}: {len(ds)} (neg={neg}, pos={pos})")

wandb.config.update(
    {
        "actual_train_size": len(splits["train"]),
        "actual_val_size": len(splits["validation"]),
        "actual_test_size": len(splits["test"]),
    }
)

# %% Cell 9: Tokenize all splits
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
    remove_columns=["content", "title"],
    desc="Tokenizing",
)
tokenized.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label"]
)

print("Tokenized dataset:")
print(tokenized)
print("\nSample tensor shapes:")
sample = tokenized["train"][0]
for k, v in sample.items():
    print(f"  {k}: {v.shape}")

# %% Cell 10: Metrics for Trainer
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }


# %% Cell 11: Initialize model
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
)
model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

wandb.config.update(
    {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }
)

# %% Cell 12: Training configuration
from transformers import TrainingArguments

LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
BATCH_SIZE = 16
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
OUTPUT_DIR = "./checkpoints/distilbert-sentiment"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=100,
    report_to="wandb",
    run_name=wandb.run.name,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    save_total_limit=2,
    seed=SEED,
)

wandb.config.update(
    {
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "fp16": torch.cuda.is_available(),
    }
)

# %% Cell 13: Train the model
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)

print("Starting fine-tuning...")
print(f"Training samples: {len(tokenized['train'])}")
print(f"Steps per epoch:  {len(tokenized['train']) // BATCH_SIZE}")
print(f"Total steps:      {len(tokenized['train']) // BATCH_SIZE * NUM_EPOCHS}")

train_result = trainer.train()

print("\nTraining complete!")
print(f"Training loss:    {train_result.training_loss:.4f}")
print(f"Training runtime: {train_result.metrics['train_runtime']:.0f}s")

# %% Cell 14: Full test set evaluation
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

test_output = trainer.predict(tokenized["test"])
test_logits = test_output.predictions
test_labels = test_output.label_ids
test_preds = np.argmax(test_logits, axis=-1)
test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()

# Classification Report
print("Test Set Classification Report:")
print(classification_report(test_labels, test_preds, target_names=LABEL_NAMES))

# ROC-AUC
roc_auc = roc_auc_score(test_labels, test_probs[:, 1])
print(f"ROC-AUC: {roc_auc:.4f}")

# Log confusion matrix to W&B
wandb.log(
    {
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_labels.tolist(),
            preds=test_preds.tolist(),
            class_names=LABEL_NAMES,
        ),
        "test/roc_auc": roc_auc,
        "test/accuracy": (test_preds == test_labels).mean(),
    }
)

# Log error analysis table to W&B
wrong_mask = test_preds != test_labels
wrong_indices = np.where(wrong_mask)[0][:50]
test_texts = [splits["test"][int(i)]["content"] for i in wrong_indices]
error_table = wandb.Table(
    columns=["text", "true_label", "predicted", "confidence"]
)
for idx in wrong_indices:
    error_table.add_data(
        test_texts[list(wrong_indices).index(idx)][:200],
        LABEL_NAMES[test_labels[idx]],
        LABEL_NAMES[test_preds[idx]],
        float(test_probs[idx].max()),
    )
wandb.log({"test/error_analysis": error_table})

# Assert minimum thresholds
test_accuracy = (test_preds == test_labels).mean()
test_f1 = f1_score(test_labels, test_preds, average="macro")

print(f"\n{'='*50}")
print("FINAL METRICS")
print(f"{'='*50}")
print(f"Accuracy: {test_accuracy:.4f} (required: >= 0.92)")
print(f"F1 Macro: {test_f1:.4f}  (required: >= 0.91)")

assert test_accuracy >= 0.92, f"Accuracy {test_accuracy:.4f} below threshold 0.92"
assert test_f1 >= 0.91, f"F1 {test_f1:.4f} below threshold 0.91"
print("\n✅ All thresholds met. Proceeding to model save.")

# %% Cell 15: Save model + tokenizer
import json

MODEL_SAVE_PATH = "./models/distilbert-sentiment"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

# Save training args for reproducibility
training_args_dict = training_args.to_dict()
with open(f"{MODEL_SAVE_PATH}/training_args.json", "w") as f:
    json.dump(training_args_dict, f, indent=2, default=str)

print(f"Model saved to: {MODEL_SAVE_PATH}")
print("Files:")
for fname in os.listdir(MODEL_SAVE_PATH):
    size_mb = os.path.getsize(f"{MODEL_SAVE_PATH}/{fname}") / 1e6
    print(f"  {fname}: {size_mb:.1f} MB")

# %% Cell 16: Write model card
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

# %% Cell 17: Log model as W&B Artefact
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
    },
)
artifact.add_dir(MODEL_SAVE_PATH)
wandb.log_artifact(artifact, aliases=["latest", "v1"])

wandb.finish()
print("✅ W&B run complete. Model artefact logged.")

# %% Cell 18: Verify inference before deploying
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
    if len(review) > 70:
        print(f'  "{review[:70]}..."')
    else:
        print(f'  "{review}"')

# %% Cell 19: Zip for download
import shutil

shutil.make_archive("distilbert-sentiment", "zip", ".", "models/distilbert-sentiment")
print("Created distilbert-sentiment.zip — download via Files panel →")
