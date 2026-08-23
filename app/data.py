"""Loading the dataset with its header corrected.

Kept separate from train.py so that anything needing the data -- training,
statistics, a future notebook -- goes through the same corrected loader
instead of re-applying the fix by hand.
"""

from __future__ import annotations

import pandas as pd

from app import config


def load_dataset(drop_duplicates: bool = True) -> pd.DataFrame:
    """Read credit_card_fraud.csv with correct column names and a 0/1 target.

    The CSV on disk is never modified. Its shipped header is discarded and
    replaced with `config.CANONICAL_COLUMNS` -- see config.py for why.
    """
    if not config.CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.CSV_PATH}. "
            "Expected credit_card_fraud.csv inside the CreditCardFraud folder."
        )

    # skiprows=1 discards the shipped header; names= supplies the correct one.
    df = pd.read_csv(
        config.CSV_PATH,
        skiprows=1,
        names=config.CANONICAL_COLUMNS,
    )

    unknown = set(df[config.TARGET_COLUMN].unique()) - set(config.LABEL_MAP)
    if unknown:
        raise ValueError(f"Unexpected labels in the target column: {unknown}")

    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map(config.LABEL_MAP)

    if drop_duplicates:
        # 10 byte-identical rows ship in the file. Left in, a duplicate can
        # land on both sides of the train/test split and inflate the score.
        df = df.drop_duplicates().reset_index(drop=True)

    return df
