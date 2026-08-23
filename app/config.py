"""Paths, column names, and the one place that knows the CSV header is wrong."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "credit_card_fraud.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
STATS_PATH = ARTIFACTS_DIR / "stats.json"

# ---------------------------------------------------------------------------
# The header shift
# ---------------------------------------------------------------------------
# credit_card_fraud.csv ships with a header row that is off by one column. The
# file is left untouched on disk; the correction happens here and nowhere else.
#
# What the header claims:  V1, V2 ... V29, time, fraud
# What the columns hold:   Time, V1 ... V28, Amount, Class
#
# Verified two ways:
#   1. Column "V1" ranges 0..172768 with 25,762 distinct values -- that is
#      seconds across a 48-hour window, i.e. Time. Column "time" ranges
#      0..8360 with a mean of 86.76 -- that is money, i.e. Amount.
#   2. Ranking columns by class separation puts V15/V13/V5/V12/V11/V18 on top.
#      Shift each name down by one and you get V14/V12/V4/V11/V10/V17, the
#      known top discriminators of this dataset.
CSV_HEADER_AS_SHIPPED = (
    ["V%d" % i for i in range(1, 30)] + ["time", "fraud"]
)

CANONICAL_COLUMNS = (
    ["Time"] + ["V%d" % i for i in range(1, 29)] + ["Amount", "Class"]
)

# The 28 PCA components plus the transaction amount.
#
# `Time` is deliberately excluded from the model. It is seconds elapsed since
# the first transaction in a fixed 48-hour collection window, so a live
# transaction would always fall far outside the trained range and a caller has
# no meaningful value to send. Set INCLUDE_TIME = True (and retrain) to put it
# back; train.py reports metrics either way.
INCLUDE_TIME = False

PCA_FEATURES = ["V%d" % i for i in range(1, 29)]
FEATURE_COLUMNS = (["Time"] if INCLUDE_TIME else []) + PCA_FEATURES + ["Amount"]

TARGET_COLUMN = "Class"

# The label column stores strings, not 0/1.
LABEL_MAP = {"fraud": 1, "otherwise": 0}

RANDOM_STATE = 42
TEST_SIZE = 0.2
