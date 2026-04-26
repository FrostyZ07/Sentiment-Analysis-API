"""Async batch job endpoints for large-scale prediction."""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.inference import predict_batch

router = APIRouter()
DB_PATH = "./data/jobs.db"


def init_jobs_db():
    """Initialize the jobs SQLite database."""
    Path(DB_PATH).parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT,
            created_at TEXT,
            completed_at TEXT,
            total INTEGER,
            results TEXT
        )
    """
    )
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
        "INSERT INTO jobs (id, status, created_at, total) "
        "VALUES (?, 'pending', ?, ?)",
        (job_id, datetime.now(timezone.utc).isoformat(), len(body.texts)),
    )
    conn.commit()
    conn.close()

    bundle = getattr(request.app.state, "bundle_v1", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    async def process_job():
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE jobs SET status='processing' WHERE id=?", (job_id,)
        )
        conn.commit()
        try:
            results = predict_batch(body.texts, bundle)
            conn.execute(
                "UPDATE jobs SET status='complete', results=?, "
                "completed_at=? WHERE id=?",
                (
                    json.dumps(results),
                    datetime.now(timezone.utc).isoformat(),
                    job_id,
                ),
            )
        except Exception as e:
            conn.execute(
                "UPDATE jobs SET status='failed', results=? WHERE id=?",
                (json.dumps({"error": str(e)}), job_id),
            )
        conn.commit()
        conn.close()

    background_tasks.add_task(process_job)
    return {"job_id": job_id, "status": "pending", "total": len(body.texts)}


@router.get("/jobs/{job_id}", summary="Get Job Status")
async def get_job_status(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT id, status, created_at, completed_at, total "
        "FROM jobs WHERE id=?",
        (job_id,),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": row[0],
        "status": row[1],
        "created_at": row[2],
        "completed_at": row[3],
        "total": row[4],
    }


@router.get("/jobs/{job_id}/results", summary="Get Job Results")
async def get_job_results(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT status, results FROM jobs WHERE id=?", (job_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")
    if row[0] != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Job status is '{row[0]}', not complete.",
        )
    return {"job_id": job_id, "results": json.loads(row[1])}
