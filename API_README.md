# Credit Card Fraud Detection API

A REST API that scores credit card transactions for fraud, wrapping the linear
SVM from `CreditCardFraud_SVM.ipynb`.

## Quick start

**The trained model ships with this repo**, so a fresh clone can serve
predictions immediately — there is no training step to run first.

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Run both from this directory, then open http://127.0.0.1:8000/docs.

Retraining is optional. `python train.py` rebuilds `artifacts/` from the CSV —
run it if you change the features, the model, or the threshold, or if you want
the model built against your own scikit-learn version.

Then open **http://127.0.0.1:8000/docs** — an interactive page, generated from
the code, where every endpoint can be tried in the browser with a real
fraudulent transaction pre-filled as the example.

## Layout

```
credit_card_fraud.csv the dataset (left exactly as it shipped)
train.py              offline: CSV -> trained pipeline -> artifacts/
app/
  config.py           paths, feature list, and the CSV header correction
  data.py             the only place that reads the CSV
  schemas.py          request/response shapes; validation lives here
  model.py            loads artifacts once at startup, scores transactions
  main.py             routes
artifacts/            committed, ~9 KB total; rebuilt by train.py
  model.joblib        scaler + classifier frozen together
  metadata.json       what was trained, when, and how it scored
  stats.json          dataset summary served by GET /stats
```

Training and serving are separate programs. `train.py` runs once and is slow;
the API runs forever and is fast. The API never trains, never reads the CSV,
and never writes to `artifacts/`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | What this service is |
| `GET` | `/health` | Liveness; `model_loaded` tells you if `/predict` will work |
| `POST` | `/predict` | Score one transaction |
| `POST` | `/predict/batch` | Score 1–1000 transactions in one call |
| `GET` | `/model/info` | Model type, training details, held-out metrics |
| `GET` | `/stats` | Summary of the training dataset |
| `GET` | `/docs` | Interactive documentation |

### `POST /predict`

Request — V1…V28 plus `Amount`. `Time` is accepted but ignored, so raw dataset
rows can be posted unchanged.

```json
{ "V1": -3.600544, "V2": 4.519047, "...": "...", "V28": 0.835749, "Amount": 3.79 }
```

Response:

```json
{
  "is_fraud": true,
  "fraud_probability": 0.999647,
  "threshold": 0.078656,
  "risk_level": "critical"
}
```

The raw probability is always returned alongside the verdict, so a caller that
wants a stricter or looser cut-off can ignore `is_fraud` and apply its own.

### Status codes

| Code | Means |
|---|---|
| `200` | Worked |
| `404` | No such path |
| `422` | Your request body failed validation — missing field, wrong type, negative `Amount`, unknown field, batch outside 1–1000 |
| `503` | Model not loaded — run `train.py` and restart |

Validation runs before the model is touched, and the `422` body names the exact
field that failed.

## Decisions worth knowing

**The CSV header is off by one.** `credit_card_fraud.csv` ships with a header
row that does not line up with its columns: the column labelled `V1` is really
`Time`, `V2…V29` are really `V1…V28`, and the column labelled `time` is really
`Amount`. The file on disk is left untouched; the correction is applied on load
in `app/config.py`, which explains how it was verified. The notebooks still read
the file as before.

**The scaler is bundled with the model.** `artifacts/model.joblib` is a
scikit-learn `Pipeline` holding both. This is not cosmetic: the notebook calls
`scaler.fit_transform(X_test)`, refitting on test data, and that operation is
impossible at serving time — a single transaction has no mean or standard
deviation of its own. Inside a pipeline, `.predict()` can only apply the scaling
learned during training.

**`Time` is excluded from the model.** It is seconds elapsed within a fixed
48-hour collection window, so a live transaction would always fall outside the
trained range and a caller has no meaningful value to send. Set
`INCLUDE_TIME = True` in `config.py` and retrain to put it back.

**The threshold is tuned for recall.** As the project README notes, a missed
fraud costs more than a false alarm. The cut-off (0.0787, well below the usual
0.5) is chosen on a validation split to maximise F2, which weights recall twice
as heavily as precision. At the default 0.5 the same model catches half as many
frauds — `GET /model/info` reports both.

**Splits are stratified and duplicate rows are dropped.** The notebook takes
`.head(36)` and `.tail(14)` of 49 fraud rows — 50 slots for 49 rows, so one
fraud transaction lands in both train and test. `train.py` uses stratified
splits instead, and drops the 10 byte-identical rows in the file.

## Results

Held out from both training and threshold selection (5,694 rows, 10 fraud):

| Metric | At threshold 0.0787 | At default 0.5 |
|---|---|---|
| Precision | 0.833 | 0.833 |
| Recall | 1.000 | 0.500 |
| F1 | 0.909 | 0.625 |
| ROC-AUC | 0.9999 | 0.9999 |
| PR-AUC | 0.881 | 0.881 |

Read `recall = 1.000` with care: **the test set contains only 10 fraud cases**,
so that figure carries a wide error bar. Scoring all 49 frauds in the file
gives 40 caught and 9 missed (~82% recall) with zero false alarms on 951
legitimate rows — but most of those rows were seen during training, so the real
number sits somewhere between the two. Accuracy is reported only to be
discounted: flagging nothing at all scores 99.83% on this data.

## Before this goes anywhere real

- **No authentication.** Anyone who can reach the port can call it. Add an API
  key or OAuth.
- **CORS is wide open** (`allow_origins=["*"]` in `main.py`). Narrow it to your
  frontend.
- **No rate limiting.**
- **Trained on 49 fraud examples** from two days in 2013. That is enough for a
  project, not for money.
- **Run without `--reload`** in production, behind a real process manager.
