"""FR-01: obtain the approved dataset and validate its schema before analysis."""
import argparse
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy.io import arff

from .config import (ALLOWED_VALUES, DATA_DIR, DATA_PATH, FEATURES, MIN_PREDICTORS,
                     MIN_RECORDS, RAW_TARGET, SEED, UCI_URL)
from .utils import get_logger

log = get_logger()


class SchemaError(ValueError):
    """Raised when a dataset does not match the expected schema."""


def download_uci_dataset(dest=DATA_PATH) -> pd.DataFrame:
    """Download the UCI Phishing Websites dataset (CC BY 4.0) and save it as CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading dataset from %s", UCI_URL)
    req = urllib.request.Request(UCI_URL, headers={"User-Agent": "fake-website-detection-student-project"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        arff_name = next(n for n in zf.namelist() if n.lower().endswith(".arff"))
        text = zf.read(arff_name).decode("utf-8", errors="replace")
    data, _ = arff.loadarff(io.StringIO(text))
    df = pd.DataFrame(data)
    for col in df.columns:
        df[col] = df[col].apply(lambda v: int(v.decode() if isinstance(v, bytes) else v))
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df.to_csv(dest, index=False)
    log.info("Saved %d rows x %d cols to %s", *df.shape, dest)
    return df


def make_synthetic_dataset(n: int = 3000, seed: int = SEED) -> pd.DataFrame:
    """SYNTHETIC data for smoke tests / offline demos only (never report as real results)."""
    rng = np.random.default_rng(seed)
    y = rng.choice([-1, 1], size=n, p=[0.45, 0.55])
    strength = rng.uniform(0.2, 0.9, size=len(FEATURES))
    cols = {}
    for f, s in zip(FEATURES, strength):
        agree = rng.random(n) < (0.5 + 0.4 * s)
        noise = rng.choice([-1, 0, 1], size=n)
        cols[f] = np.where(agree, y, noise)
    df = pd.DataFrame(cols)
    df[RAW_TARGET] = y
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Fail loudly (and log) if the dataset is not usable."""
    missing_cols = [c for c in FEATURES + [RAW_TARGET] if c not in df.columns]
    if missing_cols:
        log.error("Schema error: missing columns %s", missing_cols)
        raise SchemaError(f"Missing columns: {missing_cols}")
    if len(df) < MIN_RECORDS:
        raise SchemaError(f"Need at least {MIN_RECORDS} records, found {len(df)}")
    if len(FEATURES) < MIN_PREDICTORS:
        raise SchemaError(f"Need at least {MIN_PREDICTORS} predictors")
    bad = {}
    for c in FEATURES:
        vals = pd.to_numeric(df[c], errors="coerce")
        invalid = df[c].notna() & ~vals.isin(ALLOWED_VALUES)
        if invalid.any():
            bad[c] = int(invalid.sum())
    if bad:
        log.error("Schema error: values outside %s: %s", ALLOWED_VALUES, bad)
        raise SchemaError(f"Values outside {ALLOWED_VALUES}: {bad}")
    if not set(df[RAW_TARGET].dropna().unique()) <= {-1, 1}:
        raise SchemaError(f"{RAW_TARGET} must contain only -1 / 1")
    if df[RAW_TARGET].isna().any():
        raise SchemaError("Target column contains missing values")
    log.info("Schema OK: %d records, %d predictors", len(df), len(FEATURES))


def fix_label_polarity(df: pd.DataFrame) -> pd.DataFrame:
    """The code assumes Result == -1 means phishing and 1 means legitimate.

    Some copies of this dataset use the opposite coding. We check it from the data itself:
    legitimate sites mostly have a valid HTTPS certificate (SSLfinal_State == 1), phishing sites
    mostly do not. If the class coded -1 has the MORE valid certificates, the labels are reversed
    and we flip them (and log it, so it can be reported).
    """
    ssl_class_pos = (df.loc[df[RAW_TARGET] == 1, "SSLfinal_State"] == 1).mean()
    ssl_class_neg = (df.loc[df[RAW_TARGET] == -1, "SSLfinal_State"] == 1).mean()
    log.info("Valid-certificate share: Result==1 -> %.3f, Result==-1 -> %.3f", ssl_class_pos, ssl_class_neg)
    if ssl_class_neg > ssl_class_pos + 0.2:
        log.warning("Result labels look REVERSED in this file (phishing coded as 1). Flipping them.")
        df = df.assign(**{RAW_TARGET: -df[RAW_TARGET]})
    return df


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data_loader` to download it, "
            "or add --synthetic for an offline smoke-test dataset.")
    df = pd.read_csv(path)
    validate_schema(df)
    return fix_label_polarity(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download / create the dataset CSV")
    ap.add_argument("--synthetic", action="store_true", help="create synthetic data (smoke test only)")
    args = ap.parse_args()
    if args.synthetic:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        make_synthetic_dataset().to_csv(DATA_PATH, index=False)
        log.warning("Wrote SYNTHETIC dataset to %s - do not report results from it", DATA_PATH)
    else:
        download_uci_dataset()
    validate_schema(pd.read_csv(DATA_PATH))