"""Single training run called by W&B sweep agent.

Usage:
    wandb sweep sweeps/sweep_config.yaml --project sentiment-analysis-distilbert
    wandb agent <sweep-id> --count 10
"""
import os

import evaluate
import numpy as np
import torch
import wandb
from datasets import load_dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
TRAIN_SAMPLES = 50_000  # Smaller for sweeps
VAL_SAMPLES = 10_000

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(
        predictions=predictions, references=labels
    )
    f1_macro = f1_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )
    return {
        "accuracy": accuracy["accuracy"],
        "f1_macro": f1_macro["f1"],
    }


def train():
    with wandb.init() as run:
        config = wandb.config
        set_seed(SEED)

        # Load and sample dataset
        dataset = load_dataset("amazon_polarity")
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

        def tokenize_fn(batch):
            return tokenizer(
                batch["content"],
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
            )

        train_ds = (
            dataset["train"]
            .shuffle(seed=SEED)
            .select(range(TRAIN_SAMPLES))
        )
        val_ds = (
            dataset["test"]
            .shuffle(seed=SEED)
            .select(range(VAL_SAMPLES))
        )

        train_ds = train_ds.map(tokenize_fn, batched=True, batch_size=1000)
        val_ds = val_ds.map(tokenize_fn, batched=True, batch_size=1000)

        train_ds = train_ds.remove_columns(["content", "title"])
        val_ds = val_ds.remove_columns(["content", "title"])
        train_ds.set_format("torch")
        val_ds.set_format("torch")

        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label={0: "negative", 1: "positive"},
            label2id={"negative": 0, "positive": 1},
        )

        training_args = TrainingArguments(
            output_dir=f"./checkpoints/sweep-{run.id}",
            num_train_epochs=2,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=32,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=100,
            report_to="wandb",
            fp16=torch.cuda.is_available(),
            save_total_limit=1,
            seed=SEED,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )

        trainer.train()


if __name__ == "__main__":
    train()
