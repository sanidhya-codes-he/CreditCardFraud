"""The API itself: routes, startup, and error handling.

Start it with:

    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for an interactive page where every
endpoint below can be tried in the browser.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app import __version__, config
from app.model import fraud_model
from app.schemas import (
    BatchPrediction,
    BatchRequest,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    Prediction,
    StatsResponse,
    Transaction,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on startup, before the first request is accepted.

    The model is loaded here rather than inside a handler so that the cost is
    paid once per process instead of once per call.

    A failure to load is logged but does not stop the server: /health then
    reports model_loaded=false, which is far easier to diagnose than a process
    that refuses to boot.
    """
    try:
        fraud_model.load()
    except Exception as exc:  # noqa: BLE001 - surfaced through /health
        logger.error("Model failed to load: %s", exc)
        logger.error("Run `python train.py` to produce the artifacts.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Scores credit card transactions for fraud using a linear SVM trained "
        "on anonymised PCA features.\n\n"
        "**About the data.** Fraud is 0.17% of transactions, so accuracy is "
        "meaningless here -- flagging nothing at all would score 99.83%. The "
        "decision threshold is tuned to favour catching fraud over avoiding "
        "false alarms, because a missed fraud costs more than a review.\n\n"
        "`/predict` always returns the raw probability alongside the verdict, "
        "so a caller that wants a stricter or looser cut-off can apply its own."
    ),
    version=__version__,
    lifespan=lifespan,
    responses={503: {"model": ErrorResponse, "description": "Model unavailable"}},
)

# Allows a browser-based frontend on another port to call this API. Narrow
# allow_origins to your real frontend before putting this on the internet.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_model() -> None:
    """503 rather than a 500 stack trace when the artifacts are missing."""
    if not fraud_model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Run `python train.py`, then restart.",
        )


@app.get("/", tags=["meta"], summary="What this service is")
def root() -> dict:
    return {
        "service": "Credit Card Fraud Detection API",
        "version": __version__,
        "docs": "/docs",
        "endpoints": {
            "GET /health": "liveness and model status",
            "POST /predict": "score one transaction",
            "POST /predict/batch": "score up to 1000 transactions",
            "GET /model/info": "model type, training details, test metrics",
            "GET /stats": "summary of the training dataset",
        },
    }


@app.get("/health", tags=["meta"], response_model=HealthResponse,
         summary="Liveness check")
def health() -> HealthResponse:
    """Cheap enough to poll. Returns 200 even when the model is missing --
    the server is up either way; `model_loaded` is what tells you if /predict
    will work."""
    return HealthResponse(
        status="ok" if fraud_model.is_loaded else "degraded",
        model_loaded=fraud_model.is_loaded,
        api_version=__version__,
    )


@app.post("/predict", tags=["prediction"], response_model=Prediction,
          summary="Score one transaction")
def predict(transaction: Transaction) -> Prediction:
    """Score a single transaction.

    The body must carry V1..V28 and Amount. Anything missing, non-numeric, or
    misspelled is rejected with 422 before the model is touched.
    """
    _require_model()
    return fraud_model.predict([transaction])[0]


@app.post("/predict/batch", tags=["prediction"], response_model=BatchPrediction,
          summary="Score many transactions")
def predict_batch(request: BatchRequest) -> BatchPrediction:
    """Score 1-1000 transactions in one call.

    Far cheaper than the equivalent number of single calls: one network
    round-trip, and the model scores the whole batch as a single matrix
    operation.
    """
    _require_model()
    predictions = fraud_model.predict(request.transactions)
    return BatchPrediction(
        predictions=predictions,
        count=len(predictions),
        flagged=sum(p.is_fraud for p in predictions),
    )


@app.get("/model/info", tags=["meta"], response_model=ModelInfoResponse,
         summary="What model is running")
def model_info() -> ModelInfoResponse:
    """Which model is loaded, how it was trained, and how it scored on data it
    never saw during training."""
    _require_model()
    meta = fraud_model.metadata
    return ModelInfoResponse(
        model_type=meta["model_type"],
        trained_at=meta["trained_at"],
        features=meta["features"],
        n_features=meta["n_features"],
        threshold=meta["threshold"],
        training=meta["training"],
        metrics=meta["metrics"],
    )


@app.get("/stats", tags=["meta"], response_model=StatsResponse,
         summary="Training dataset summary")
def stats() -> StatsResponse:
    """Dataset summary, precomputed at training time so this stays instant."""
    _require_model()
    return StatsResponse(**fraud_model.stats)
