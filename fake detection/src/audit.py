"""Checkpoint 2: data audit (missingness, duplicates, balance, quality risks)."""
import json

import pandas as pd

from .config import FEATURES, RAW_TARGET, REPORTS_DIR


def run_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Write audit tables/JSON to reports/ and return the de-duplicated frame."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    n_before = len(df)
    n_dups = int(df.duplicated().sum())
    clean = df.drop_duplicates().reset_index(drop=True)

    balance = clean[RAW_TARGET].value_counts(normalize=True).to_dict()
    near_const = [c for c in FEATURES if clean[c].value_counts(normalize=True).iloc[0] > 0.95]
    table = pd.DataFrame({
        "feature": FEATURES,
        "missing": [int(df[c].isna().sum()) for c in FEATURES],
        "n_unique": [int(df[c].nunique()) for c in FEATURES],
        "most_common_share": [round(float(df[c].value_counts(normalize=True).iloc[0]), 3) for c in FEATURES],
    })
    table.to_csv(REPORTS_DIR / "data_audit_table.csv", index=False)

    risks = [
        f"Duplicate rows: {n_dups} exact duplicates removed BEFORE splitting so identical "
        "records cannot appear in both train and test (leakage).",
        f"Class balance (legit/fake share): {({int(k): round(v, 3) for k, v in balance.items()})} - "
        "accuracy alone can mislead, so precision/recall/F1 on the fake class are reported.",
        f"Near-constant features (>95% one value): {near_const or 'none'} - carry little signal.",
        "Pre-encoded features (-1/0/1) came from a fixed extraction recipe on one collection "
        "period; they may drift as phishing tactics evolve.",
    ]
    audit = {
        "rows_before": n_before, "duplicates_removed": n_dups, "rows_after": len(clean),
        "missing_total": int(df[FEATURES + [RAW_TARGET]].isna().sum().sum()),
        "class_share": {str(k): v for k, v in balance.items()},
        "quality_risks": risks,
    }
    (REPORTS_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return clean