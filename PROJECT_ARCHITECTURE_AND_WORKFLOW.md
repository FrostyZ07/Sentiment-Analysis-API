# Sentiment Analysis API: Architecture, Workflow, and Current State

## 1. High-Level Architecture Overview

This project is an end-to-end MLOps solution that serves a fine-tuned DistilBERT model for sentiment analysis. The architecture is divided into four distinct phases: **Training**, **CI/CD Automation**, **Serving**, and **Monitoring**.

```mermaid
flowchart TD
    subgraph Training Phase
        A[Hugging Face Datasets: amazon_polarity] --> B(Data Preprocessing)
        B --> C[DistilBERT Fine-Tuning]
        C -- Logs Metrics --> D[(Weights & Biases)]
        C -- Pushes Model --> E[(Hugging Face Hub)]
    end

    subgraph CI/CD Automation
        F[GitHub Repository] -- Push/PR --> G{GitHub Actions}
        G -- ci.yml --> H[Lint, Test, Docker Smoke]
        G -- deploy.yml --> I[Build & Push to GHCR]
        I --> J[Deploy to Railway]
    end

    subgraph Serving Phase
        K[FastAPI Application]
        K -- Loads Model on Startup --> E
        L[Client Request] --> M[Middleware: Auth, Rate Limit, CORS]
        M --> N{Endpoints}
        N -- /predict --> O[Sync Inference]
        N -- /batch --> P[Micro-batch Inference]
        N -- /jobs/batch --> Q[(SQLite Jobs DB)]
        Q -. Async Processing .-> P
    end

    subgraph Monitoring Phase
        R[Prometheus Metrics] <-- Exposes /metrics -- K
        S[Evidently AI] <-- Analyzes SQLite request logs -- K
        S -- Generates --> T[Data Drift HTML Reports]
    end
```

---

## 2. Component Behavior & Responsibilities

### 2.1 Backend Core (`app/`)
*   **`main.py`**: The FastAPI entry point. Manages the application lifecycle (loading the HuggingFace model into memory on startup via `lifespan`), wires up middlewares, and includes all routers.
*   **`core/model.py`**: Handles downloading and loading the `DistilBertForSequenceClassification` model and tokenizer. Wraps them in a `ModelBundle` dataclass.
*   **`core/middleware.py`**: Handles API key validation and tiered rate limiting. Unauthenticated users get strict limits; authenticated users get higher limits.
*   **`core/config.py`**: Uses `pydantic-settings` to load environment variables and configuration (e.g., model path, API keys, CORS allowed origins).
*   **`core/metrics.py`**: Sets up Prometheus `Counter` and `Histogram` objects to track request volume, inference latency, and confidence score distribution.

### 2.2 API Routes (`app/api/v1/routes/`)
*   **`predict.py`**: Handles synchronous single-review predictions. Cleans text, runs tokenization, and returns the sentiment label and confidence score.
*   **`batch.py`**: Handles synchronous batch predictions (up to 32 items). Uses micro-batching under the hood for GPU/CPU efficiency.
*   **`jobs.py`**: Handles large-scale asynchronous batch prediction. Returns a `job_id` immediately, processes the batch in a background task, and stores results in a local SQLite database (`data/jobs.db`) to be retrieved via polling.
*   **`health.py`**: Exposes `/health` (liveness probe) and `/ready` (readiness probe checking if the ML model is fully loaded in memory).

### 2.3 Services & Inference (`app/services/`)
*   **`inference.py`**: Contains the pure PyTorch execution logic. Runs the forward pass (`torch.no_grad()`), applies softmax, extracts probabilities, and records Prometheus metrics.
*   **`drift.py`**: Logs every incoming prediction request to a local SQLite database (`data/request_log.db`). Uses **Evidently AI** to compare the text length and word count distributions of recent live traffic against the original training dataset to detect Data Drift.

### 2.4 DevOps & Deployment (`docker/`, `.github/`)
*   **`Dockerfile`**: A multi-stage build. Uses a slim builder stage to compile C-extensions, then copies them to a runtime stage running as a non-root user. During the build, it triggers `scripts/download_model.sh` to bake the HuggingFace model directly into the image to prevent slow startups and timeout errors on Railway.
*   **CI Workflow (`ci.yml`)**: Runs on every push/PR. Executes Ruff, Black, and isort for linting. Runs Pytest with a minimum 60% coverage gate. Builds a test Docker image and runs a smoke test against the `/health` endpoint.
*   **CD Workflow (`deploy.yml`)**: Runs on merges to `main`. Builds the multi-platform Docker image, pushes it to GitHub Container Registry (GHCR), and triggers a deployment on Railway using the Railway CLI.

---

## 3. The Exact Request Workflow (e.g., `/predict`)

1.  **Client makes a POST request** to `https://<railway-domain>/api/v1/predict` with a JSON payload containing the review text.
2.  **Middleware Processing**:
    *   *Request ID*: A unique UUID is assigned to the request for traceability.
    *   *CORS*: Validates the Origin header.
    *   *Rate Limiter*: Checks the IP address or API key against the `slowapi` limits.
3.  **Pydantic Validation**: FastAPI parses the payload through `PredictRequest` in `app/schemas/request.py`. It ensures the text is not empty and is between 3 and 2000 characters.
4.  **Route Handler**: `predict_single()` is called. It retrieves the pre-loaded `ModelBundle` from `app.state`.
5.  **Inference Service**:
    *   HTML tags and extra whitespaces are stripped.
    *   The text is tokenized (padded/truncated to 128 tokens).
    *   Passed through the DistilBERT model.
    *   Logits are converted to probabilities using Softmax.
6.  **Telemetry & Monitoring**:
    *   The prediction latency and confidence score are recorded by Prometheus.
    *   The input text length and predicted label are logged to the Drift DB.
7.  **Response**: The JSON response is returned containing the sentiment (`positive`/`negative`), confidence score, and processing time.

---

## 4. Current State of the Project

**Status:** **Release Candidate / Final CI/CD Polish**

*   **Machine Learning**: **COMPLETE**. The model was trained on the `amazon_polarity` dataset, achieved >92% accuracy, and is successfully hosted on Hugging Face at `FrostyZ07/distilbert-sentiment-amazon`.
*   **Backend Application**: **COMPLETE**. The FastAPI application is fully functional, all endpoints (including async batch and drift detection) are implemented.
*   **Test Suite**: **COMPLETE**. 23 pytest cases are passing with ~62% coverage, successfully mocking the ML model to run fast without a GPU.
*   **Deployment Configuration**: **COMPLETE**. The Dockerfile correctly bakes the model into the image, avoiding the Railway "502 Bad Gateway" timeout errors that happen if models download dynamically at runtime.
*   **CI/CD Pipeline**: **IN PROGRESS (Final Fixes)**. 
    *   We recently fixed issues where PyTorch thread limits caused silent OOM kills.
    *   We fixed the Hugging Face download script to correctly point to the `distilbert-sentiment` subfolder within the Hugging Face repository.
    *   *Immediate next step*: Resolving a strict `black` formatting error in the GitHub Actions CI pipeline caused by Windows (CRLF) vs Linux (LF) line-ending mismatches. We have added a `.gitattributes` file and renormalized the line endings to unblock the CI pipeline. Once the CI pipeline turns green, the final image will be pushed to GHCR and deployed to Railway automatically.
