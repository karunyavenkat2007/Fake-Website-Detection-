# Fake Website Detection Using Machine Learning

Cybernaut Open-Choice ML Project (Brief 01, Route A: supervised learning).

**Problem:** Given 30 pre-extracted indicators of a website (URL, page, domain features), predict whether it is a
fake/phishing site (`1`) or legitimate (`0`). Expected output: a label plus a fake-probability a user can act on.
**Dataset:** UCI *Phishing Websites* (id 327), CC BY 4.0, ~11k records, 30 predictors, no passwords/personal data.
**Metric:** F1 on the fake class (plus precision/recall, ROC-AUC). **Failure =** not beating the majority baseline, or
high false-negative rate on any subgroup.

## Setup in VS Code
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run (reproducible, seed = 42)
```bash
python -m src.data_loader          # download + validate dataset (FR-01)
python -m src.train                # audit, split, baseline + 2 models, tuning, reports, save model (FR-02..06)
python -m src.predict --file sample_inputs/fake_example.json    # new-record prediction (validated)
python -m src.predict --file sample_inputs/legit_example.json
python -m pytest -q                # schema, leakage and validation tests
streamlit run app.py               # web app
```
Offline smoke test only: `python -m src.data_loader --synthetic` (never report results from synthetic data).

## Requirement map
| SRS | Where |
|---|---|
| FR-01 schema validation | `src/data_loader.py::validate_schema` |
| FR-02 feature types, target, missing values, transforms | `src/config.py`, `src/audit.py`, `src/modeling.py` |
| FR-03 baseline + 2 algorithms | majority baseline, Logistic Regression, Random Forest (`src/modeling.py`) |
| FR-04 confusion matrix / class metrics | `reports/confusion_matrices.png`, `model_comparison.csv`, `classification_report_best.txt` |
| FR-05 >=2 error slices + new-record prediction | `reports/error_slices.csv` (4 slices), `worst_errors.csv`, `src/predict.py` |
| FR-06 model artifact + run command | `models/fake_website_model.joblib`, commands above |
| Leakage control | duplicates removed before split; imputer/scaler live inside the pipeline; tuning by CV on train only; model chosen by CV score, test set used once; proven in `tests/test_project.py` |

## Deploy (Streamlit Community Cloud, free)
1. `git init`, commit everything (including `models/` and `reports/` after training), push to a GitHub repo.
2. Go to share.streamlit.io, create an app from the repo, main file `app.py`.
3. If `models/` is absent the app downloads the data and trains on first start.

Alternative: `docker build -t fakesite . && docker run -p 8501:8501 fakesite` (works on Render, Railway, etc.).

## Known limitations
- Features are pre-encoded (-1/0/1) from a fixed 2015-era recipe, so the app takes indicators, not a raw URL.
  Extracting these features from a live URL (WHOIS, page crawl) is the natural improvement.
- Phishing tactics drift; retrain on fresh data before real-world use. This is a learning project, not a security product.