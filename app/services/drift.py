"""Input drift detection using Evidently AI."""
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = "./data/request_log.db"
REPORTS_DIR = Path("./reports")
MAX_LOG_SIZE = 1000  # Keep last N requests


def init_db():
    """Initialize SQLite database for request logging."""
    Path(DB_PATH).parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS request_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            text_length INTEGER,
            word_count INTEGER,
            predicted_label TEXT,
            confidence REAL
        )
    """
    )
    conn.commit()
    conn.close()


def log_request(text: str, sentiment: str, confidence: float):
    """Log a prediction request to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO request_log (timestamp, text_length, word_count,
                                 predicted_label, confidence)
        VALUES (?, ?, ?, ?, ?)
    """,
        (
            datetime.utcnow().isoformat(),
            len(text),
            len(text.split()),
            sentiment,
            confidence,
        ),
    )
    # Trim to MAX_LOG_SIZE
    conn.execute(
        f"""
        DELETE FROM request_log WHERE id NOT IN (
            SELECT id FROM request_log ORDER BY id DESC LIMIT {MAX_LOG_SIZE}
        )
    """
    )
    conn.commit()
    conn.close()


def run_drift_check(reference_stats_path: str = "./data/training_stats.json"):
    """
    Compare recent API traffic stats against training distribution.
    Logs warning if drift detected.
    """
    try:
        import pandas as pd
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        conn = sqlite3.connect(DB_PATH)
        current_df = pd.read_sql(
            "SELECT * FROM request_log ORDER BY id DESC LIMIT 500", conn
        )
        conn.close()

        if len(current_df) < 100:
            logger.info(
                "drift.skip: insufficient data, count=%d", len(current_df)
            )
            return

        with open(reference_stats_path) as f:
            ref_data = json.load(f)

        ref_df = pd.DataFrame(ref_data)

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=ref_df,
            current_data=current_df[["text_length", "word_count"]],
        )

        REPORTS_DIR.mkdir(exist_ok=True)
        report_path = (
            REPORTS_DIR / f"drift_{datetime.utcnow().strftime('%Y%m%d')}.html"
        )
        report.save_html(str(report_path))

        drift_detected = report.as_dict()["metrics"][0]["result"][
            "dataset_drift"
        ]
        if drift_detected:
            logger.warning("drift.detected, report_path=%s", str(report_path))
        else:
            logger.info(
                "drift.none_detected, report_path=%s", str(report_path)
            )

    except ImportError:
        logger.warning("drift.skip: evidently not installed")
    except Exception as e:
        logger.error("drift.check_failed: %s", str(e))
