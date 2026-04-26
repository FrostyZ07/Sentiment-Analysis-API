# Sentiment Analysis API

[![CI](https://github.com/FrostyZ07/Sentiment-Analysis-API/actions/workflows/ci.yml/badge.svg)](https://github.com/FrostyZ07/Sentiment-Analysis-API/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fine-tuned DistilBERT model for product review sentiment classification,
deployed as a production-grade REST API. End-to-end MLOps project demonstrating
the complete model lifecycle from data ingestion to live serving.

**API Docs:** `/docs` (Swagger UI)
**W&B Dashboard:** [wandb.ai/thanmayshetty/sentiment-analysis-distilbert](https://wandb.ai/thanmayshetty/sentiment-analysis-distilbert)

---

## Architecture

```
┌──────────┐     ┌─────────────────────────────────┐     ┌──────────┐
│  Client  │────▶│  FastAPI + Uvicorn (async)       │────▶│DistilBERT│
│ (curl /  │     │  ├─ CORS Middleware              │     │  Model   │
│  browser │◀────│  ├─ Rate Limiter (slowapi)       │◀────│ (PyTorch)│
│  / app)  │     │  ├─ Request ID + Logging         │     └──────────┘
└──────────┘     │  ├─ Prometheus /metrics          │
                 │  └─ API Key Auth (optional)      │
                 └─────────────────────────────────┘
```

## Model Performance

| Metric | Target |
|--------|--------|
| Accuracy | >= 92% |
| F1 Macro | >= 91% |
| ROC-AUC | >= 95% |

## Quickstart

### Local (Python)

```bash
git clone https://github.com/FrostyZ07/Sentiment-Analysis-API.git
cd Sentiment-Analysis-API
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt
# Place model in models/distilbert-sentiment/ first
uvicorn app.main:app --reload
```

### Local (Docker)

```bash
docker build -f docker/Dockerfile -t sentiment-api:latest .
docker run -p 8000:8000 -v ./models:/app/models:ro sentiment-api:latest
```

## API Usage

### Single Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Best product I have ever bought!", "return_probabilities": true}'
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible quality, broke immediately."]}'
```

### Async Batch Job

```bash
# Submit
curl -X POST http://localhost:8000/api/v1/jobs/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["review1", "review2", ...]}'

# Poll status
curl http://localhost:8000/api/v1/jobs/{job_id}

# Get results
curl http://localhost:8000/api/v1/jobs/{job_id}/results
```

## Reproduce Training

1. Open `notebooks/train_sentiment_distilbert.py` in Google Colab
2. Set `WANDB_API_KEY` in Colab secrets
3. Run all cells — training takes ~2.5 hours on T4 GPU
4. Download model from W&B artefact or HF Hub

## Stand-Out Features

- **W&B Hyperparameter Sweep** — Bayesian optimization across 10 runs
- **Prometheus Metrics** — Custom ML metrics at `/metrics`
- **API Key Auth** — Tiered rate limits (60 vs 300 req/min)
- **Drift Detection** — Daily Evidently AI reports
- **Async Batch Jobs** — Up to 1000 texts per job with polling

## CI/CD

Every PR runs: lint (ruff + black + isort) -> tests (pytest + coverage) -> Docker smoke test
Every merge to main: build multi-platform image -> push to GHCR -> deploy to Railway

## Project Structure

```
├── app/                    # FastAPI application
│   ├── api/v1/routes/      # Endpoint handlers
│   ├── core/               # Config, model loader, middleware
│   ├── schemas/            # Pydantic request/response models
│   └── services/           # Inference, drift detection
├── tests/                  # pytest test suite
├── docker/                 # Dockerfile + compose
├── scripts/                # Download, sweep, load test
├── notebooks/              # Training notebook
├── monitoring/             # Prometheus + Grafana
├── .github/workflows/      # CI/CD pipelines
└── sweeps/                 # W&B sweep config
```

## License

MIT — see [LICENSE](LICENSE)
