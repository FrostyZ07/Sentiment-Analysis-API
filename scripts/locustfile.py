"""Locust load test for Sentiment Analysis API."""
import random

from locust import HttpUser, between, task

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
    wait_time = between(0.1, 0.5)  # 0.1-0.5s between requests

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
