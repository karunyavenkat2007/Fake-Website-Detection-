"""Run:  python -m pytest -q   (uses synthetic data, so it works offline)"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.config import FEATURES, RAW_TARGET, SEED
from src.data_loader import SchemaError, make_synthetic_dataset, validate_schema
from src.modeling import build_logreg
from src.predict import ValidationError, validate_record

GOOD = {f: 1 for f in FEATURES}


def test_schema_accepts_valid_and_rejects_bad():
    df = make_synthetic_dataset(600)
    validate_schema(df)
    with pytest.raises(SchemaError):
        validate_schema(df.drop(columns=[FEATURES[0]]))
    bad = df.copy()
    bad.loc[0, FEATURES[1]] = 7
    with pytest.raises(SchemaError):
        validate_schema(bad)
    with pytest.raises(SchemaError):
        validate_schema(df.head(100))


def test_preprocessing_fitted_on_train_only():
    df = make_synthetic_dataset(1500).drop_duplicates()
    X, y = df[FEATURES].astype(float), (df[RAW_TARGET] == -1).astype(int)
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pipe = build_logreg().fit(X_tr, y_tr)
    scaler = pipe.named_steps["scale"]
    assert np.allclose(scaler.mean_, X_tr.mean().values)
    assert not np.allclose(scaler.mean_, X.mean().values)   # would differ if fitted on all data


def test_no_duplicate_rows_across_split():
    df = make_synthetic_dataset(1500).drop_duplicates()
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED)
    assert len(pd.merge(tr, te, how="inner")) == 0


def test_validation_errors_are_clear():
    validate_record(GOOD)
    with pytest.raises(ValidationError, match="Missing"):
        validate_record({k: v for k, v in GOOD.items() if k != "SFH"})
    with pytest.raises(ValidationError, match="text"):
        validate_record({**GOOD, "SFH": "abc"})
    with pytest.raises(ValidationError, match="one of"):
        validate_record({**GOOD, "SFH": 5})
    with pytest.raises(ValidationError, match="whole number"):
        validate_record({**GOOD, "SFH": 0.5})
    with pytest.raises(ValidationError, match="Unknown"):
        validate_record({**GOOD, "colour": 1})


def test_allow_missing_returns_nan_row():
    df = validate_record({k: v for k, v in GOOD.items() if k != "SFH"}, allow_missing=True)
    assert df["SFH"].isna().all()