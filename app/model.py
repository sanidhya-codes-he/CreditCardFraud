"""Loading the trained artifacts and scoring transactions.

The model is read from disk exactly once, when the server process starts, and
then held in memory for the lifetime of the process. Loading it per request
would add hundreds of milliseconds to every call for no benefit.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from joblib import load

from app import config
from app.schemas import Prediction, Transaction

logger = logging.getLogger(__name__)

# Boundaries for the human-readable risk label. The verdict itself always
# comes from the trained threshold; these are only for display.
RISK_BANDS = ((0.25, "low"), (0.50, "medium"), (0.75, "high"))


class FraudModel:
    """The trained pipeline plus everything the API needs to describe it."""

    def __init__(self) -> None:
        self.pipeline = None
        self.metadata: dict = {}
        self.stats: dict = {}
        self.threshold: float = 0.5

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def load(self) -> None:
        """Read artifacts/ into memory. Raises if training has not been run."""
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No model at {config.MODEL_PATH}. Run `python train.py` first."
            )

        self.pipeline = load(config.MODEL_PATH)
        self.metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
        self.stats = json.loads(config.STATS_PATH.read_text(encoding="utf-8"))
        self.threshold = float(self.metadata.get("threshold", 0.5))

        logger.info(
            "Loaded %s (trained %s, threshold %.4f)",
            self.metadata.get("model_type", "model"),
            self.metadata.get("trained_at", "unknown"),
            self.threshold,
        )

    def _to_frame(self, transactions: list[Transaction]) -> pd.DataFrame:
        """Turn validated requests into the exact column order used in training.

        Building a DataFrame with named columns rather than a bare array is
        deliberate: it makes a mismatch between the request and the trained
        feature set impossible to get wrong silently.
        """
        frame = pd.DataFrame([t.model_dump() for t in transactions])
        if "Time" in config.FEATURE_COLUMNS:
            # Time is optional in the request but required by the model when
            # config.INCLUDE_TIME is on; absent means "start of window".
            frame["Time"] = frame["Time"].fillna(0.0)
        return frame[config.FEATURE_COLUMNS]

    def predict(self, transactions: list[Transaction]) -> list[Prediction]:
        """Score transactions. The whole batch goes through the model at once."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        # .predict_proba applies the scaling learned during training. There is
        # no fitting here, and there could not be -- a single transaction has
        # no mean or standard deviation of its own.
        probabilities = self.pipeline.predict_proba(self._to_frame(transactions))[:, 1]

        return [
            Prediction(
                is_fraud=bool(p >= self.threshold),
                fraud_probability=round(float(p), 6),
                threshold=self.threshold,
                risk_level=self._risk_level(p),
            )
            for p in probabilities
        ]

    @staticmethod
    def _risk_level(probability: float | np.floating) -> str:
        for bound, label in RISK_BANDS:
            if probability < bound:
                return label
        return "critical"


# One instance shared by every request handler.
fraud_model = FraudModel()
