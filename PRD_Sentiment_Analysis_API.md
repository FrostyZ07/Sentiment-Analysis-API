# Product Requirements Document
## Sentiment Analysis API — Fine-Tuned DistilBERT on Amazon Reviews

**Version:** 1.0  
**Author:** MLOps Project Team  
**Date:** April 2026  
**Status:** Active  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Market Context & Opportunity](#2-market-context--opportunity)
3. [Project Goals & Success Metrics](#3-project-goals--success-metrics)
4. [Stakeholders & Target Audience](#4-stakeholders--target-audience)
5. [Technical Architecture Overview](#5-technical-architecture-overview)
6. [Deliverable 1 — Training Notebook with W&B Logging](#6-deliverable-1--training-notebook-with-wb-logging)
7. [Deliverable 2 — FastAPI Endpoint with Documentation](#7-deliverable-2--fastapi-endpoint-with-documentation)
8. [Deliverable 3 — GitHub Repo with CI/CD via Actions](#8-deliverable-3--github-repo-with-cicd-via-actions)
9. [Data Requirements](#9-data-requirements)
10. [Model Requirements](#10-model-requirements)
11. [API Specification](#11-api-specification)
12. [Non-Functional Requirements](#12-non-functional-requirements)
13. [Stand-Out & Scalability Features](#13-stand-out--scalability-features)
14. [Risk Register](#14-risk-register)
15. [Timeline & Milestones](#15-timeline--milestones)
16. [Appendix — Technology Stack Reference](#16-appendix--technology-stack-reference)

---

## 1. Executive Summary

This document defines all product, technical, and quality requirements for the **Sentiment Analysis API** — an end-to-end MLOps project that fine-tunes DistilBERT on the Amazon Reviews dataset (Hugging Face) and serves predictions through a production-grade REST API built with FastAPI.

The project is designed as a complete MLOps learning artefact that demonstrates the full model lifecycle: data ingestion → experimentation → fine-tuning → experiment tracking → containerised deployment → CI/CD automation → scalable serving.

Beyond the base deliverables, the PRD maps several stand-out enhancements that raise the project from a portfolio piece to a genuine, marketable ML service — including multi-class star-rating prediction, confidence scoring, batch endpoints, model versioning, drift monitoring, and a freemium API key tier.

---

## 2. Market Context & Opportunity

### 2.1 Market Size

The global MLOps market was valued at **USD 1.7 billion in 2024** and is projected to grow at a **CAGR of 37.4%** through 2034, driven by enterprise demand for reproducible, scalable AI workflows. Concurrently, the LLM fine-tuning orchestration segment is accelerating adoption of platforms that provide built-in audit trails, version control, and governance dashboards — requirements this project directly addresses.

### 2.2 Sentiment Analysis Demand

Sentiment analysis is among the highest-demand NLP applications in 2025–2026:

- **E-commerce** platforms use it to aggregate review signals at scale.
- **Brand monitoring** tools integrate it for real-time social listening.
- **Financial services** use domain-fine-tuned variants for earnings call analysis.
- **Customer success** teams route tickets using predicted sentiment polarity.

Traditional rule-based tools are being rapidly replaced by transformer-based solutions. Research shows that fine-tuned DistilBERT achieves **90%+ accuracy** on Amazon review datasets — competitive with much larger models while remaining cost-efficient to serve.

### 2.3 Competitive Gap

Most existing open-source sentiment APIs lack:

- Confidence intervals on predictions
- Multi-class star-rating (1–5) support beyond binary positive/negative
- Batch prediction endpoints for bulk review processing
- Model versioning and A/B rollout endpoints
- Live drift monitoring against training distribution

This PRD specifies all of these as stand-out requirements in Section 13.

---

## 3. Project Goals & Success Metrics

### 3.1 Primary Goals

| Goal | Description |
|------|-------------|
| G-01 | Demonstrate complete MLOps cycle from raw data to deployed prediction API |
| G-02 | Achieve ≥ 92% binary accuracy and ≥ 0.91 F1-score on held-out test set |
| G-03 | Deliver a publicly accessible REST API with < 300 ms p99 latency at baseline load |
| G-04 | Implement full experiment tracking with W&B for reproducibility |
| G-05 | Automate testing and deployment with GitHub Actions CI/CD pipeline |

### 3.2 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Model Accuracy (binary) | ≥ 92% | Test set evaluation in training notebook |
| F1-Score | ≥ 0.91 (macro) | W&B logged evaluation run |
| API p50 Latency | < 100 ms | Load test (Locust) at 10 RPS |
| API p99 Latency | < 300 ms | Load test (Locust) at 10 RPS |
| CI Pipeline Pass Rate | 100% on `main` branch | GitHub Actions status |
| API Uptime | ≥ 99.5% | UptimeRobot / Railway health checks |
| OpenAPI Docs Coverage | 100% endpoints documented | Auto-generated Swagger UI inspection |
| W&B Run Reproducibility | Training re-run within ± 0.5% accuracy | Manual re-run verification |

---

## 4. Stakeholders & Target Audience

### 4.1 Project Stakeholders

| Role | Responsibilities |
|------|-----------------|
| ML Engineer (owner) | Model training, fine-tuning, experiment tracking |
| API Developer (owner) | FastAPI implementation, endpoint design, documentation |
| DevOps / MLOps (owner) | CI/CD pipeline, containerisation, deployment |
| Reviewers / Evaluators | Portfolio assessment, code review |

### 4.2 API Consumer Personas

**Persona A — Developer / Indie Hacker**  
Needs a simple REST endpoint to add sentiment tagging to an e-commerce app. Cares about documentation quality, latency, and free-tier availability.

**Persona B — Data Science Team**  
Needs batch prediction for offline analysis of 10k–100k reviews. Cares about throughput, confidence scores, and CSV/JSON output.

**Persona C — Enterprise Integration**  
Needs API key authentication, SLA guarantees, versioned endpoints, and a usage dashboard. Cares about reliability, data privacy, and compliance logging.

---

## 5. Technical Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TRAINING PHASE                              │
│                                                                       │
│  HuggingFace Datasets  ──►  Data Preprocessing  ──►  DistilBERT     │
│  (amazon_polarity)           (tokenisation,           Fine-Tuning    │
│                               EDA, splits)            + W&B Logging  │
│                                                             │         │
│                                                     Model Registry   │
│                                                     (HF Hub / W&B)   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          SERVING PHASE                               │
│                                                                       │
│  GitHub Repo  ──►  GitHub Actions CI/CD  ──►  Docker Image          │
│  (source)         (lint, test, build)          (FastAPI + model)     │
│                                                         │             │
│                                              Railway / Replit Deploy │
│                                                         │             │
│                                              REST API (FastAPI)       │
│                                              /predict  /batch         │
│                                              /health   /docs          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MONITORING PHASE                              │
│                                                                       │
│  Prometheus Metrics  ──►  Grafana Dashboard  ──►  Drift Alerts      │
│  (latency, errors,         (optional)              (evidently-ai)    │
│   request counts)                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Core technology decisions:**

- **Model:** `distilbert-base-uncased` — 40% smaller than BERT, 60% faster inference, retains 97% of BERT's performance on GLUE benchmarks
- **Dataset:** `amazon_polarity` (HuggingFace) — 3.6 million training samples, binary (positive / negative) labels
- **Training framework:** Hugging Face `Trainer` API with `transformers` + `datasets`
- **Experiment tracking:** Weights & Biases (W&B) — runs, metrics, model artefacts
- **Serving framework:** FastAPI with Uvicorn ASGI server
- **Containerisation:** Docker (multi-stage build, slim Python 3.11 base)
- **Deployment:** Railway (primary) or Replit (secondary / demo)
- **CI/CD:** GitHub Actions

---

## 6. Deliverable 1 — Training Notebook with W&B Logging

### 6.1 Scope

A Jupyter notebook (`notebooks/train_sentiment_distilbert.ipynb`) that walks through the complete fine-tuning pipeline end-to-end. It must be fully reproducible — re-running all cells in order must produce a model within ± 0.5% of the original accuracy.

### 6.2 Notebook Sections (Required)

#### Section 0: Environment & Seed Setup
- Pin all library versions in `requirements-train.txt`
- Set global random seed (`torch`, `numpy`, `random`, `transformers.set_seed`)
- Log GPU/CPU environment info to W&B run config

#### Section 1: Dataset Loading & Exploratory Data Analysis (EDA)
- Load `amazon_polarity` from HuggingFace `datasets`
- Display class distribution (must be logged as W&B bar chart)
- Display review length distribution (histogram — W&B `wandb.plot.histogram`)
- Show sample reviews per class
- Identify and flag token length outliers (> 512 tokens before truncation)

#### Section 2: Data Preprocessing
- Tokenise with `DistilBertTokenizer` (`distilbert-base-uncased`)
- Apply padding and truncation to `max_length=128` (validated against length distribution)
- Create train / validation / test splits (80% / 10% / 10%)
- Log split sizes and class balance per split to W&B config

#### Section 3: Model Initialisation
- Load `DistilBertForSequenceClassification` with `num_labels=2`
- Log model parameter count to W&B
- Display model architecture summary

#### Section 4: Training Configuration
- Define `TrainingArguments`:
  - `learning_rate`: 2e-5 (logged as hyperparameter)
  - `num_train_epochs`: 3
  - `per_device_train_batch_size`: 16
  - `warmup_ratio`: 0.1
  - `weight_decay`: 0.01
  - `evaluation_strategy`: "epoch"
  - `report_to`: "wandb"
- All hyperparameters surfaced as W&B config keys

#### Section 5: Training Execution
- Use HuggingFace `Trainer` with `compute_metrics` function
- Log the following per epoch to W&B:
  - Training loss
  - Validation loss
  - Accuracy
  - F1-score (macro and weighted)
  - Precision
  - Recall
  - ROC-AUC (binary)
- Log best model checkpoint path

#### Section 6: Test Set Evaluation
- Evaluate best checkpoint on held-out test set
- Generate and log:
  - Classification report (`sklearn`)
  - Confusion matrix (as W&B `wandb.plot.confusion_matrix`)
  - ROC curve (as W&B custom chart)
  - Sample incorrect predictions (logged as W&B Table for error analysis)

#### Section 7: Model Saving & Artefact Logging
- Save model and tokenizer locally (`./models/distilbert-sentiment/`)
- Log saved model directory as W&B Artefact (`type="model"`)
- Push model to HuggingFace Hub (optional, script-controlled via env var `HF_PUSH=true`)
- Generate `model_card.md` with performance summary

#### Section 8: Inference Demo
- Run live inference on 5 custom review strings
- Print prediction + confidence score (softmax probability)

### 6.3 W&B Integration Requirements

| W&B Feature | Usage |
|-------------|-------|
| `wandb.init()` | Project: `sentiment-analysis-distilbert`, entity: configurable via env |
| `wandb.config` | All hyperparameters, dataset sizes, model name |
| `wandb.log()` | All epoch-level metrics |
| `wandb.plot.*` | Loss curves, confusion matrix, ROC curve, class distribution |
| `wandb.Table` | Error analysis samples, test set predictions |
| `wandb.Artifact` | Saved model checkpoint |
| W&B Sweeps | Hyperparameter sweep config (YAML) — see Stand-Out Section 13.1 |

### 6.4 Acceptance Criteria

- [ ] Notebook runs end-to-end without errors on Google Colab (T4 GPU) or local GPU
- [ ] Test set accuracy ≥ 92%
- [ ] Test set F1-score (macro) ≥ 0.91
- [ ] W&B run is publicly viewable (link included in `README.md`)
- [ ] All 8 sections present and functional
- [ ] Model artefact logged to W&B with metadata

---

## 7. Deliverable 2 — FastAPI Endpoint with Documentation

### 7.1 Scope

A production-grade FastAPI application (`app/`) that loads the fine-tuned DistilBERT model at startup and serves predictions through versioned REST endpoints with automatic OpenAPI documentation.

### 7.2 Project Structure

```
app/
├── main.py                  # FastAPI app factory, lifespan, router includes
├── api/
│   ├── v1/
│   │   ├── routes/
│   │   │   ├── predict.py   # Single prediction endpoint
│   │   │   ├── batch.py     # Batch prediction endpoint
│   │   │   └── health.py    # Health + readiness checks
│   │   └── __init__.py
│   └── __init__.py
├── core/
│   ├── config.py            # Pydantic Settings (env vars)
│   ├── model.py             # Model loader singleton
│   └── middleware.py        # Rate limiting, request ID, CORS
├── schemas/
│   ├── request.py           # Pydantic request models
│   └── response.py          # Pydantic response models
├── services/
│   └── inference.py         # Tokenisation + inference logic
└── tests/
    ├── test_predict.py
    ├── test_batch.py
    └── test_health.py
```

### 7.3 Endpoints

#### GET `/health`
**Purpose:** Liveness probe — confirms the service process is running.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2026-04-25T10:00:00Z"
}
```

#### GET `/ready`
**Purpose:** Readiness probe — confirms model is loaded and ready to serve.

**Response (200):**
```json
{
  "status": "ready",
  "model": "distilbert-sentiment-v1",
  "model_loaded": true,
  "timestamp": "2026-04-25T10:00:00Z"
}
```

**Response (503 — model not loaded):**
```json
{
  "status": "not_ready",
  "model_loaded": false
}
```

#### POST `/api/v1/predict`
**Purpose:** Single review sentiment prediction.

**Request body:**
```json
{
  "text": "This product exceeded all my expectations. Truly remarkable build quality.",
  "return_probabilities": true
}
```

**Request schema constraints:**
- `text`: string, min length 3, max length 2000 characters
- `return_probabilities`: boolean, default `false`

**Response (200):**
```json
{
  "sentiment": "positive",
  "label_id": 1,
  "confidence": 0.9847,
  "probabilities": {
    "negative": 0.0153,
    "positive": 0.9847
  },
  "processing_time_ms": 42.3,
  "model_version": "distilbert-sentiment-v1"
}
```

**Error Responses:**
- `422 Unprocessable Entity` — validation failure (Pydantic)
- `429 Too Many Requests` — rate limit exceeded
- `500 Internal Server Error` — inference failure

#### POST `/api/v1/batch`
**Purpose:** Batch prediction for multiple reviews in a single request.

**Request body:**
```json
{
  "texts": [
    "Great product, fast shipping.",
    "Completely fell apart after two days.",
    "Average quality, nothing special."
  ],
  "return_probabilities": false
}
```

**Request schema constraints:**
- `texts`: list of strings, min 1 item, max 32 items per request
- Each string: min 3 chars, max 2000 chars

**Response (200):**
```json
{
  "results": [
    {
      "index": 0,
      "text_preview": "Great product, fast shipp...",
      "sentiment": "positive",
      "confidence": 0.978
    },
    {
      "index": 1,
      "text_preview": "Completely fell apart aft...",
      "sentiment": "negative",
      "confidence": 0.992
    },
    {
      "index": 2,
      "text_preview": "Average quality, nothing ...",
      "sentiment": "negative",
      "confidence": 0.623
    }
  ],
  "total": 3,
  "processing_time_ms": 95.1,
  "model_version": "distilbert-sentiment-v1"
}
```

#### GET `/api/v1/model/info`
**Purpose:** Returns model metadata.

**Response:**
```json
{
  "model_name": "distilbert-sentiment-v1",
  "base_model": "distilbert-base-uncased",
  "dataset": "amazon_polarity",
  "labels": ["negative", "positive"],
  "test_accuracy": 0.9231,
  "test_f1_macro": 0.9228,
  "training_date": "2026-04-20",
  "wandb_run_url": "https://wandb.ai/..."
}
```

### 7.4 Middleware Requirements

| Middleware | Specification |
|------------|--------------|
| CORS | Allow configurable origins via `ALLOWED_ORIGINS` env var; default `*` for dev |
| Rate Limiting | 60 requests/minute per IP (anonymous); implemented via `slowapi` |
| Request ID | Inject `X-Request-ID` UUID header on every response |
| Logging | Structured JSON logs (timestamp, request_id, method, path, status, latency_ms) |
| Error Handling | Global exception handler returns consistent JSON error schema |

### 7.5 Model Loading

- Model loaded **once at startup** using FastAPI lifespan context manager (not deprecated `@app.on_event`)
- Model stored as application state (`app.state.model`, `app.state.tokenizer`)
- If model files not found, app raises `RuntimeError` and fails to start (readiness probe returns 503)
- Model path configurable via `MODEL_PATH` environment variable
- Support loading from local path OR HuggingFace Hub model ID

### 7.6 Documentation Requirements

- Swagger UI auto-generated at `/docs`
- ReDoc at `/redoc`
- Every endpoint must have:
  - `summary` and `description` in route decorator
  - Request and response examples embedded in Pydantic schemas via `model_config` / `json_schema_extra`
  - Documented error responses (422, 429, 500)
- `README.md` in `app/` directory with quickstart curl examples for all endpoints

### 7.7 Acceptance Criteria

- [ ] All 5 endpoints return correct responses for valid inputs
- [ ] Pydantic validation rejects invalid inputs with 422 + descriptive message
- [ ] Rate limiter returns 429 after threshold breach
- [ ] Swagger UI accessible at `/docs` with all endpoints documented
- [ ] Model loaded from path set in `MODEL_PATH` env var
- [ ] `GET /ready` returns 503 before model is loaded
- [ ] Response always includes `model_version` and `processing_time_ms`
- [ ] Structured logs emitted for every request

---

## 8. Deliverable 3 — GitHub Repo with CI/CD via Actions

### 8.1 Repository Structure

```
sentiment-api/
├── .github/
│   └── workflows/
│       ├── ci.yml          # Lint + test on every PR
│       └── deploy.yml      # Build + push Docker image on merge to main
├── app/                    # FastAPI application (see Section 7)
├── notebooks/              # Training notebook (see Section 6)
├── data/                   # Data processing scripts (not raw data)
│   └── prepare_dataset.py
├── models/                 # Gitignored; model downloaded at runtime
│   └── .gitkeep
├── scripts/
│   ├── download_model.sh   # Downloads model from HF Hub or W&B
│   └── run_load_test.sh    # Locust load test wrapper
├── docker/
│   ├── Dockerfile          # Multi-stage production Docker image
│   └── docker-compose.yml  # Local development compose
├── tests/
│   ├── conftest.py
│   ├── test_predict.py
│   ├── test_batch.py
│   └── test_health.py
├── requirements.txt        # API runtime dependencies (pinned)
├── requirements-train.txt  # Training dependencies (pinned)
├── requirements-dev.txt    # Linting / testing dependencies
├── pyproject.toml          # Black, isort, ruff config
├── .env.example            # Template env file (no secrets)
├── model_card.md           # Auto-generated by training notebook
└── README.md               # Full project documentation
```

### 8.2 README Requirements

The project `README.md` must include:

1. **Project overview** — one-paragraph description, badges (CI status, license, Python version)
2. **Architecture diagram** — ASCII or Mermaid diagram of full MLOps pipeline
3. **Quickstart** — `docker-compose up` or `uvicorn` one-liner to run locally
4. **API usage** — curl examples for all endpoints
5. **Training** — step-by-step to reproduce fine-tuning from scratch
6. **W&B dashboard link** — public run URL
7. **Model performance** — table of test set metrics
8. **Deployment** — how to deploy to Railway / Replit
9. **CI/CD** — description of GitHub Actions pipelines
10. **Stand-out features** — description of enhancements beyond base requirements
11. **Contributing** — PR guide, code style, test requirements
12. **License** — MIT

### 8.3 CI Pipeline (`ci.yml`)

**Trigger:** Push to any branch, Pull Request to `main`

**Jobs:**

**Job 1: `lint`**
- Checkout code
- Set up Python 3.11
- Install `ruff`, `black`, `isort`
- Run `ruff check .` — fail on any lint error
- Run `black --check .` — fail on formatting issues
- Run `isort --check-only .`

**Job 2: `test`** (depends on `lint`)
- Checkout code
- Set up Python 3.11
- Cache pip dependencies
- Install `requirements.txt` + `requirements-dev.txt`
- Download or mock model (see below)
- Run `pytest tests/ -v --cov=app --cov-report=xml`
- Upload coverage report to Codecov
- Fail if coverage < 80%

**Model mocking strategy for CI:**
- Tests use `pytest` fixtures with `unittest.mock.patch` to mock `app.state.model` and `app.state.tokenizer`
- A lightweight `conftest.py` fixture replaces the full model with a mock that returns deterministic predictions
- This eliminates the need to download the 250 MB model during every CI run

**Job 3: `docker-build`** (depends on `test`)
- Build Docker image (`docker build`)
- Run smoke test: start container, call `GET /health`, assert 200

### 8.4 CD Pipeline (`deploy.yml`)

**Trigger:** Push to `main` branch (after merge)

**Jobs:**

**Job 1: `build-and-push`**
- Checkout code
- Set up Docker Buildx
- Log in to GitHub Container Registry (GHCR)
- Build multi-platform image (`linux/amd64`, `linux/arm64`)
- Push image tagged with `latest` and `sha-{commit_hash}`

**Job 2: `deploy-railway`** (depends on `build-and-push`)
- Install Railway CLI
- Deploy to Railway using `RAILWAY_TOKEN` secret
- Wait for deployment health check to pass
- Post deployment status to PR comment (using `github-script`)

### 8.5 Docker Requirements

**Dockerfile (multi-stage):**

```
Stage 1 (builder): python:3.11-slim
  - Install build dependencies
  - Install pip packages into /install

Stage 2 (runtime): python:3.11-slim
  - Copy /install from builder
  - Copy app/ source
  - Run as non-root user (uid 1000)
  - Expose port 8000
  - CMD: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

**Image requirements:**
- Final image size < 1.5 GB (model excluded; downloaded at runtime via `download_model.sh`)
- Non-root user execution
- `.dockerignore` excludes notebooks, tests, raw data, `__pycache__`

### 8.6 Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `MODEL_PATH` | Local path or HF Hub ID for model | Yes | `./models/distilbert-sentiment` |
| `WANDB_API_KEY` | W&B API key for logging | No | — |
| `WANDB_PROJECT` | W&B project name | No | `sentiment-analysis` |
| `HF_TOKEN` | HuggingFace token for private models | No | — |
| `ALLOWED_ORIGINS` | CORS allowed origins (comma-separated) | No | `*` |
| `RATE_LIMIT_PER_MINUTE` | Requests per minute per IP | No | `60` |
| `LOG_LEVEL` | Logging level | No | `INFO` |
| `PORT` | Server port | No | `8000` |

All variables documented in `.env.example` with descriptions.

### 8.7 Acceptance Criteria

- [ ] CI pipeline passes on clean clone with no local setup
- [ ] CD pipeline deploys to Railway on every merge to `main`
- [ ] Test coverage ≥ 80% (enforced by CI)
- [ ] Docker image builds successfully and passes smoke test
- [ ] All secrets stored in GitHub repository secrets (never in source code)
- [ ] `README.md` meets all 12 content requirements from Section 8.2
- [ ] Deployed URL is publicly accessible and listed in `README.md`

---

## 9. Data Requirements

### 9.1 Primary Dataset

**Dataset:** `amazon_polarity` (HuggingFace `datasets` library)  
**Size:** 3,600,000 training samples, 400,000 test samples  
**Labels:** Binary — `0` (negative), `1` (positive)  
**Fields used:** `content` (review text), `label`  

### 9.2 Data Sampling Strategy

Full dataset training is not required for the project deliverable. The following sampling strategy is recommended for feasibility on free-tier GPUs:

| Split | Samples | Rationale |
|-------|---------|-----------|
| Train | 200,000 | ~5.5% of full train; sufficient for ≥ 92% accuracy |
| Validation | 25,000 | Balanced class evaluation |
| Test | 25,000 | Final held-out evaluation |

Sampling is stratified by label to maintain 50/50 class balance. Sampling parameters are logged to W&B config.

### 9.3 Data Processing Rules

- Strip HTML tags from review text before tokenisation
- Remove reviews with fewer than 3 words (too short to carry sentiment signal)
- Truncate at `max_length=128` tokens (covers > 95% of reviews without loss)
- No PII anonymisation required (Amazon Reviews are public data)

### 9.4 Data Versioning

- Raw dataset sourced programmatically via `datasets.load_dataset()` — no raw files committed to Git
- Dataset version pinned via `datasets` library version in `requirements-train.txt`
- Preprocessing parameters (max_length, sample sizes, splits) logged to W&B config for reproducibility

---

## 10. Model Requirements

### 10.1 Base Model

| Property | Value |
|----------|-------|
| Model ID | `distilbert-base-uncased` |
| Parameters | 66 million |
| Size on disk | ~250 MB |
| Inference speed | ~2x faster than BERT-base |
| Architecture | 6-layer transformer, 768 hidden dim |

### 10.2 Fine-Tuning Specification

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Learning rate | 2e-5 | Standard for transformer fine-tuning; prevents catastrophic forgetting |
| Epochs | 3 | Diminishing returns observed after epoch 3 on amazon_polarity |
| Batch size | 16 | Fits T4 GPU (16 GB VRAM) with `max_length=128` |
| Warmup ratio | 0.1 | Linear warmup for first 10% of steps |
| Weight decay | 0.01 | L2 regularisation to reduce overfitting |
| Optimizer | AdamW | Default for HF Trainer |
| LR scheduler | Linear decay | Default for HF Trainer |

### 10.3 Minimum Performance Thresholds

Fine-tuned model must achieve on the held-out test set:

| Metric | Threshold |
|--------|-----------|
| Accuracy | ≥ 92% |
| F1-score (macro) | ≥ 0.91 |
| Precision (positive class) | ≥ 0.90 |
| Recall (negative class) | ≥ 0.90 |
| ROC-AUC | ≥ 0.97 |

If thresholds are not met, the training notebook must re-run with adjusted hyperparameters before the model is deployed.

### 10.4 Model Artefact Requirements

The saved model directory must contain:

- `config.json` — model architecture config
- `pytorch_model.bin` or `model.safetensors` — model weights
- `tokenizer_config.json` — tokenizer config
- `vocab.txt` — DistilBERT vocabulary
- `special_tokens_map.json`
- `training_args.json` — training hyperparameters
- `model_card.md` — performance summary

---

## 11. API Specification

### 11.1 Base URL

**Production:** `https://sentiment-api-<hash>.railway.app`  
**Local development:** `http://localhost:8000`

### 11.2 Versioning

All ML inference endpoints are versioned under `/api/v1/`. Version header `X-API-Version: 1` also accepted.

Future model versions increment the path: `/api/v2/predict`.

### 11.3 Authentication (Base + Stand-out)

**Base requirement:** No authentication required for public endpoints (anonymous access).

**Stand-out extension (Section 13.3):** API key authentication via `X-API-Key` header enabling tiered rate limits.

### 11.4 Response Schema Standards

All responses follow a consistent envelope:

**Success:** HTTP 2xx with JSON body as specified per endpoint.

**Error:**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "text must be at least 3 characters",
    "request_id": "req-uuid-here",
    "timestamp": "2026-04-25T10:00:00Z"
  }
}
```

**Error codes:**
- `VALIDATION_ERROR` — 422 — invalid request input
- `RATE_LIMIT_EXCEEDED` — 429 — too many requests
- `MODEL_NOT_READY` — 503 — model not loaded
- `INFERENCE_ERROR` — 500 — prediction failure

### 11.5 Performance Targets

| Endpoint | p50 Latency | p99 Latency |
|----------|-------------|-------------|
| `POST /api/v1/predict` | < 80 ms | < 250 ms |
| `POST /api/v1/batch` (32 items) | < 400 ms | < 800 ms |
| `GET /health` | < 5 ms | < 20 ms |

Targets measured on Railway Standard instance (2 vCPU, 512 MB RAM) with model pre-loaded.

---

## 12. Non-Functional Requirements

### 12.1 Reliability

- Application must restart automatically on crash (Railway restart policy)
- Health check endpoint polled every 30 seconds; unhealthy containers restarted
- Model loaded idempotently — multiple startup calls must not cause errors

### 12.2 Security

- No secrets in source code or Docker image — all via environment variables
- Input text sanitised (strip HTML) before tokenisation
- CORS configured to restrict origins in production
- Non-root Docker user

### 12.3 Observability

- Structured JSON logs for every request (request_id, path, status, latency_ms)
- `/metrics` endpoint exposing Prometheus-compatible counters (stand-out, Section 13.4)
- W&B run URL included in model metadata response

### 12.4 Code Quality

- All Python files formatted with `black` (line length 88)
- Imports sorted with `isort`
- Linting via `ruff` (E, W, F rules at minimum)
- Type hints on all function signatures
- Docstrings on all modules, classes, and public functions

### 12.5 Testing

- Unit tests for all inference service functions
- Integration tests for all API endpoints using FastAPI `TestClient`
- Mocked model for CI speed
- Minimum 80% line coverage enforced by CI

---

## 13. Stand-Out & Scalability Features

These features go beyond the base project requirements and significantly elevate the project in portfolio reviews, demonstrating production ML engineering competency.

### 13.1 W&B Hyperparameter Sweep

**What:** Automated hyperparameter optimisation using W&B Sweeps.

**Implementation:**
- Define sweep config YAML (`sweeps/sweep_config.yaml`) with ranges for learning rate (1e-5 to 5e-5), batch size (8, 16), warmup ratio (0.05, 0.1, 0.2)
- Use Bayesian optimisation strategy to find best combination in < 10 runs
- Best run automatically tagged `best` in W&B artefact registry

**Why it stands out:** Demonstrates production experimentation discipline — not just running one training job but systematically finding optimal hyperparameters.

### 13.2 Multi-Class Star Rating Prediction (1–5)

**What:** Extend the model to predict 5-class star ratings in addition to binary sentiment.

**Implementation:**
- Fine-tune a second model head with `num_labels=5` on the same dataset (using star ratings 1–5 as targets)
- Add endpoint `POST /api/v1/predict/rating` that returns predicted star rating + confidence
- Both model heads loaded at startup; controlled via query param `?task=binary|rating`

**Why it stands out:** Binary sentiment is table stakes. Star rating prediction is commercially differentiated — directly relevant to e-commerce ranking, review summarisation, and NPS scoring pipelines.

### 13.3 API Key Authentication with Tiered Rate Limits

**What:** Add API key authentication enabling tiered access.

**Implementation:**
- API key stored as SHA-256 hash in environment variable `API_KEYS` (comma-separated)
- `X-API-Key` header checked in middleware
- Anonymous: 60 requests/minute
- Authenticated (free tier key): 300 requests/minute
- Rate limit tier stored in key metadata (simple JSON env var)
- `/api/v1/keys/validate` endpoint to check key validity

**Why it stands out:** Transforms the project from a demo into a monetisable SaaS prototype. Demonstrates understanding of API security and tiered access models — essential for any production API.

### 13.4 Prometheus Metrics + Grafana Dashboard

**What:** Expose ML-specific operational metrics beyond basic HTTP metrics.

**Implementation:**
- Integrate `prometheus-fastapi-instrumentator`
- Expose `/metrics` endpoint
- Add custom metrics:
  - `sentiment_prediction_total{label="positive|negative"}` — prediction distribution
  - `sentiment_confidence_histogram` — distribution of confidence scores
  - `inference_duration_seconds` — tokenisation + model forward pass duration
- Include `docker-compose.monitoring.yml` that spins up Prometheus + Grafana locally
- Provide pre-built Grafana dashboard JSON (importable)

**Why it stands out:** Operational ML metrics are a major gap in most portfolio projects. This demonstrates awareness of production model monitoring — a core MLOps competency.

### 13.5 Input Drift Detection with Evidently AI

**What:** Compare live API traffic text statistics against training distribution to detect drift.

**Implementation:**
- Log 1000 most recent API requests (text length, predicted label, confidence) to a rolling SQLite buffer
- Scheduled job (every 24 hours via APScheduler) runs Evidently `DataDriftPreset` comparing live vs. training distribution
- Drift report exported as HTML to `/reports/drift_{date}.html`
- If drift p-value < 0.05 on text length or prediction distribution, log warning to structured logs and W&B

**Why it stands out:** Concept drift is one of the most common causes of production ML model degradation. Implementing drift detection demonstrates senior ML engineering awareness and is a rare addition to student projects.

### 13.6 Async Batch Processing with Task Queue

**What:** For large batch requests (> 32 items), support async job submission with polling.

**Implementation:**
- `POST /api/v1/jobs/batch` — accepts up to 1000 texts, returns `job_id`
- Background task processes batch using FastAPI `BackgroundTasks`
- `GET /api/v1/jobs/{job_id}` — returns job status (`pending`, `processing`, `complete`, `failed`)
- `GET /api/v1/jobs/{job_id}/results` — returns results when complete
- Job state stored in SQLite (lightweight, no Redis dependency)

**Why it stands out:** Demonstrates understanding of async patterns in ML serving — fundamental for production systems where synchronous inference is too slow for large batches.

### 13.7 Model Versioning & A/B Testing Endpoint

**What:** Support serving two model versions simultaneously for A/B comparison.

**Implementation:**
- Load `MODEL_PATH_V1` and `MODEL_PATH_V2` at startup (V2 optional)
- `POST /api/v1/predict?version=v1|v2|ab` — `ab` randomly routes 50/50
- A/B results logged with version tag to allow offline analysis
- `/api/v1/model/compare` — returns side-by-side accuracy metrics for V1 vs V2

**Why it stands out:** Model versioning and A/B rollout is a core MLOps pattern used at every major ML platform (Seldon, BentoML, SageMaker). Including it demonstrates production deployment maturity.

---

## 14. Risk Register

| Risk ID | Risk Description | Likelihood | Impact | Mitigation |
|---------|-----------------|------------|--------|------------|
| R-01 | Free GPU quota exhausted mid-training | Medium | High | Use Google Colab Pro or Kaggle (30h/week free GPU); train on 200k sample not full dataset |
| R-02 | Model accuracy below 92% threshold | Low | High | Use validated hyperparameters from research (lr=2e-5, 3 epochs); baseline well-documented in literature |
| R-03 | Railway free tier limits memory for model | Medium | Medium | Use quantised model (INT8) via `optimum`; deploy to Replit as fallback |
| R-04 | W&B run not publicly reproducible | Low | Medium | Enable public project visibility before sharing; document API key requirement |
| R-05 | CI pipeline slow due to model download | High | Low | Mock model in tests; only download in CD pipeline |
| R-06 | Docker image > 2 GB | Medium | Low | Use multi-stage build; exclude model from image; download at runtime |
| R-07 | Drift detection false positives | Medium | Low | Set conservative p-value threshold (0.01 not 0.05); require drift on 2+ metrics |

---

## 15. Timeline & Milestones

| Week | Milestone | Key Deliverables |
|------|-----------|-----------------|
|  | Data & Training Setup | Dataset loaded, EDA complete, W&B project created, base model loading confirmed |
|  | Fine-Tuning Complete | Training notebook complete, model meets accuracy thresholds, artefact logged to W&B |
|  | FastAPI Core | Predict + health endpoints working locally, Pydantic schemas, model loading |
|  | API Complete | All endpoints, middleware, docs, rate limiting, batch endpoint done |
|  | CI/CD Setup | GitHub Actions CI pipeline passing, Docker image building, Dockerfile complete |
|  | Deployment | Deployed to Railway, CD pipeline live, public URL in README |
|  | Stand-Out Features | 2–3 stand-out features from Section 13 implemented and documented |
|  | Polish & Documentation | README complete, model card published, W&B run public, load test run |

---

## 16. Appendix — Technology Stack Reference

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `transformers` | ≥ 4.40 | DistilBERT model + tokenizer |
| `datasets` | ≥ 2.18 | Amazon Reviews dataset loading |
| `torch` | ≥ 2.2 | PyTorch backend |
| `wandb` | ≥ 0.17 | Experiment tracking |
| `fastapi` | ≥ 0.111 | REST API framework |
| `uvicorn` | ≥ 0.29 | ASGI server |
| `pydantic` | ≥ 2.7 | Request/response validation |
| `slowapi` | ≥ 0.1.9 | Rate limiting |
| `pytest` | ≥ 8.2 | Testing |
| `ruff` | ≥ 0.4 | Linting |
| `black` | ≥ 24.4 | Code formatting |
| `scikit-learn` | ≥ 1.4 | Evaluation metrics |

### Stand-Out Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `evidently` | ≥ 0.4 | Drift detection |
| `prometheus-fastapi-instrumentator` | ≥ 7.0 | Prometheus metrics |
| `apscheduler` | ≥ 3.10 | Scheduled drift jobs |
| `optimum` | ≥ 1.20 | Model quantisation (INT8) |

### Infrastructure

| Tool | Plan | Purpose |
|------|------|---------|
| Railway | Starter ($5/mo) | Primary deployment |
| Replit | Free | Secondary / demo deployment |
| GitHub Actions | Free (2000 min/mo) | CI/CD |
| GitHub Container Registry | Free | Docker image storage |
| W&B | Free (100 GB storage) | Experiment tracking |
| Google Colab | Free (T4 GPU) | Model training |

---

*This PRD is a living document. Updates should be tracked via Git commits with descriptive messages referencing the section modified. All major changes to requirements should be discussed with project stakeholders before implementation.*

---

**Document End — Sentiment Analysis API PRD v1.0**
