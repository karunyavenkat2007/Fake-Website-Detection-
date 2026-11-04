"""Streamlit web app.  Run locally:  streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

from src.config import (DATA_PATH, FEATURES, METADATA_PATH, MODEL_PATH, REPORTS_DIR)
from src.predict import ValidationError, predict_dataframe, validate_record

st.set_page_config(page_title="Fake Website Detector", page_icon="🛡️", layout="wide")

LABELS = {-1: "-1  phishing-like", 0: "0  suspicious", 1: "1  legitimate-like"}


@st.cache_resource(show_spinner="First run: preparing data and training the model...")
def ensure_model():
    """On a fresh deployment (no model file yet) download data and train once."""
    if not MODEL_PATH.exists():
        from src.data_loader import download_uci_dataset, make_synthetic_dataset
        from src.train import main as train_main
        if not DATA_PATH.exists():
            try:
                download_uci_dataset()
            except Exception:  # offline fallback so the demo still starts
                DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                make_synthetic_dataset().to_csv(DATA_PATH, index=False)
        train_main()
    return True


ensure_model()

st.title("🛡️ Fake Website Detection using Machine Learning")
st.caption("Predicts whether a website is fake/phishing from 30 pre-extracted URL, page and domain indicators "
           "(UCI Phishing Websites dataset). For learning and screening only, not a security guarantee.")

tab1, tab2, tab3 = st.tabs(["Single website", "Batch CSV", "Model report"])

with tab1:
    st.write("Set each indicator: **-1** phishing-like, **0** suspicious, **1** legitimate-like.")
    record, cols = {}, st.columns(3)
    for i, f in enumerate(FEATURES):
        with cols[i % 3]:
            record[f] = st.selectbox(f, [-1, 0, 1], index=2, format_func=LABELS.get, key=f)
    if st.button("Check website", type="primary"):
        try:
            res = predict_dataframe(validate_record(record)).iloc[0]
            p = float(res["fake_probability"])
            (st.error if p >= 0.5 else st.success)(f"{res['prediction']}  (fake probability {p:.1%})")
        except (ValidationError, FileNotFoundError) as e:
            st.warning(str(e))

with tab2:
    st.write("Upload a CSV with the 30 feature columns (values -1 / 0 / 1).")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                raise ValidationError(f"CSV is missing columns: {missing}")
            bad = df[FEATURES].isna().any(axis=1) | ~df[FEATURES].isin([-1, 0, 1]).all(axis=1)
            if bad.any():
                raise ValidationError(f"Rows with missing/invalid values: {list(df.index[bad][:10])}")
            out = predict_dataframe(df[FEATURES])
            st.dataframe(out[["prediction", "fake_probability"] + FEATURES])
            st.download_button("Download results", out.to_csv(index=False), "predictions.csv")
        except (ValidationError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            st.warning(str(e))

with tab3:
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        st.json({k: meta[k] for k in ("selected_model", "best_params", "holdout_metrics", "dataset_source", "seed")})
    for name, title in [("model_comparison.csv", "Baseline vs models (same holdout set)"),
                        ("error_slices.csv", "Subgroup / error-slice diagnostics")]:
        p = REPORTS_DIR / name
        if p.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(p))
    for img in ("confusion_matrices.png", "feature_importance.png"):
        if (REPORTS_DIR / img).exists():
            st.image(str(REPORTS_DIR / img))