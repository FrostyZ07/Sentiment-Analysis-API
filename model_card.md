# DistilBERT Sentiment Analysis — Amazon Reviews

## Model Description
Fine-tuned `distilbert-base-uncased` for binary sentiment classification
on Amazon product reviews (positive / negative).

## Training Data
- **Dataset:** amazon_polarity (HuggingFace)
- **Train samples:** 200,000
- **Validation samples:** 25,000
- **Test samples:** 25,000

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 |
| Epochs | 3 |
| Batch Size | 16 |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.01 |
| Max Token Length | 128 |

## Performance (Test Set)
*Exact values populated after training.*

| Metric | Target |
|--------|--------|
| Accuracy | >= 92% |
| F1 Macro | >= 91% |
| ROC-AUC | >= 95% |

## Labels
- `0` → negative
- `1` → positive

## Usage
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="./models/distilbert-sentiment")
result = classifier("This product is amazing!")
print(result)  # [{'label': 'positive', 'score': 0.98}]
```
