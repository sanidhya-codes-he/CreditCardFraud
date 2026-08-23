"""Request and response shapes.

Pydantic validates every incoming request against these classes before the
handler runs. A request with a missing field, a string where a float belongs,
or a negative Amount is rejected with 422 and never reaches the model.

These classes are also what FastAPI turns into the interactive docs at /docs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# A genuine fraudulent transaction from the dataset, shown as the pre-filled
# example on the /docs page so the endpoint can be tried with one click.
_EXAMPLE_FRAUD = {
    "V1": -3.600544, "V2": 4.519047, "V3": -6.340884, "V4": 6.214767,
    "V5": -5.829558, "V6": -2.478095, "V7": -9.938412, "V8": 2.830086,
    "V9": -5.659162, "V10": -11.298156, "V11": 7.252953, "V12": -14.275092,
    "V13": 0.188903, "V14": -14.555957, "V15": -0.338289, "V16": -12.146540,
    "V17": -25.162799, "V18": -9.042845, "V19": 0.787579, "V20": 1.384743,
    "V21": 2.263770, "V22": 0.620749, "V23": -0.094069, "V24": 0.536719,
    "V25": 0.398142, "V26": 0.008277, "V27": 2.053524, "V28": 0.835749,
    "Amount": 3.79,
}


class Transaction(BaseModel):
    """One credit card transaction to be scored.

    V1..V28 are the anonymised PCA components that ship with this dataset --
    the original features were transformed for confidentiality, so there is no
    real-world meaning attached to any individual one.
    """

    model_config = ConfigDict(
        extra="forbid",  # a typo'd field name is an error, not silently dropped
        json_schema_extra={"example": _EXAMPLE_FRAUD},
    )

    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")

    Amount: float = Field(
        ..., ge=0, description="Transaction amount. Must not be negative."
    )

    # Accepted so that raw dataset rows can be posted unchanged, but ignored by
    # the model -- see config.INCLUDE_TIME.
    Time: float | None = Field(
        default=None,
        description=(
            "Seconds since the first transaction in the dataset. Optional and "
            "unused by the model; accepted so raw dataset rows can be posted "
            "as-is."
        ),
    )


class BatchRequest(BaseModel):
    """A list of transactions scored in a single call."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"transactions": [_EXAMPLE_FRAUD]}}
    )

    transactions: list[Transaction] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Between 1 and 1000 transactions.",
    )


class Prediction(BaseModel):
    """The verdict on one transaction."""

    is_fraud: bool = Field(
        ..., description="True when fraud_probability >= threshold."
    )
    fraud_probability: float = Field(
        ..., ge=0, le=1, description="Model confidence that this is fraud."
    )
    threshold: float = Field(
        ..., description="Cut-off used to turn the probability into a verdict."
    )
    risk_level: str = Field(
        ..., description="One of: low, medium, high, critical."
    )


class BatchPrediction(BaseModel):
    """Verdicts for a batch, plus a count of the ones flagged."""

    predictions: list[Prediction]
    count: int = Field(..., description="Transactions scored.")
    flagged: int = Field(..., description="How many were flagged as fraud.")


class HealthResponse(BaseModel):
    """Liveness probe. `model_loaded` false means /predict will fail."""

    status: str
    model_loaded: bool
    api_version: str


class ModelInfoResponse(BaseModel):
    """What is actually running, and how well it scored when trained."""

    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    trained_at: str
    features: list[str]
    n_features: int
    threshold: float
    training: dict[str, Any]
    metrics: dict[str, Any]


class StatsResponse(BaseModel):
    """Summary of the dataset the model was trained on."""

    dataset: dict[str, Any]
    class_balance: dict[str, Any]
    amount: dict[str, Any]
    top_features: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    """Body returned with every 4xx and 5xx response."""

    detail: str
