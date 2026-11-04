"""New-record prediction with strict input validation.

Examples:
    python -m src.predict --file sample_inputs/fake_example.json
    python -m src.predict --file my_record.json --allow-missing
"""
import argparse
import json
import sys
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import ALLOWED_VALUES, FEATURES, MODEL_PATH
from .utils import get_logger

log = get_logger()


class ValidationError(ValueError):
    """Raised for missing / unknown / wrongly typed / out-of-range fields."""


@lru_cache(maxsize=1)
def load_model(path=str(MODEL_PATH)):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at {path}. Run `python -m src.train` first.")


def _coerce(name, value):
    if isinstance(value, bool) or value is None:
        raise ValidationError(f"'{name}' must be one of {ALLOWED_VALUES}, got {value!r}")
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise ValidationError(f"'{name}' must be numeric (-1, 0 or 1), got text {value!r}")
    if not isinstance(value, (int, float, np.integer, np.floating)) or float(value) != int(value):
        raise ValidationError(f"'{name}' must be a whole number in {ALLOWED_VALUES}, got {value!r}")
    if int(value) not in ALLOWED_VALUES:
        raise ValidationError(f"'{name}' must be one of {ALLOWED_VALUES}, got {int(value)}")
    return int(value)


def validate_record(record: dict, allow_missing: bool = False) -> pd.DataFrame:
    """Return a 1-row DataFrame in training column order, or raise ValidationError."""
    if not isinstance(record, dict):
        raise ValidationError("Input must be a JSON object of feature: value pairs")
    unknown = sorted(set(record) - set(FEATURES))
    if unknown:
        raise ValidationError(f"Unknown fields: {unknown}")
    missing = [f for f in FEATURES if f not in record or record[f] in ("", None)]
    if missing and not allow_missing:
        raise ValidationError(f"Missing required fields: {missing} (use --allow-missing to impute)")
    if missing:
        log.warning("Missing fields imputed with training-set mode: %s", missing)
    row = {f: (np.nan if f in missing else _coerce(f, record[f])) for f in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    proba = model.predict_proba(df[FEATURES])[:, 1]
    out = df.copy()
    out["fake_probability"] = proba.round(4)
    out["prediction"] = np.where(proba >= 0.5, "FAKE / phishing", "legitimate")
    return out


def predict_record(record: dict, allow_missing: bool = False) -> dict:
    df = validate_record(record, allow_missing)
    res = predict_dataframe(df).iloc[0]
    return {"prediction": res["prediction"], "fake_probability": float(res["fake_probability"])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="path to a JSON file with 30 features")
    src.add_argument("--json", help="inline JSON string (use a file on PowerShell)")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    try:
        text = open(args.file, encoding="utf-8").read() if args.file else args.json
        result = predict_record(json.loads(text), args.allow_missing)
        print(json.dumps(result, indent=2))
    except (ValidationError, json.JSONDecodeError, FileNotFoundError) as e:
        log.error("Prediction failed: %s", e)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)