"""Streamlit page: paste a real URL and get a prediction. Appears automatically in the app sidebar."""
import pandas as pd
import streamlit as st

from src.config import DATA_PATH
from src.url_model import URL_MODEL_PATH, predict_url, train_url_model

st.set_page_config(page_title="Check a URL", page_icon="🔎", layout="wide")
st.title("🔎 Check a real website")
st.caption("Paste a URL. The app works out the indicators it can (URL text, DNS, TLS certificate, page HTML) "
           "and predicts. The page is read as text only; it is never opened or run.")


@st.cache_resource(show_spinner="Training the URL model (first run only)...")
def ensure_url_model():
    if not URL_MODEL_PATH.exists():
        if not DATA_PATH.exists():
            from src.data_loader import download_uci_dataset
            download_uci_dataset()
        train_url_model()
    return True


try:
    ensure_url_model()
except Exception as e:
    st.error(f"Model not ready: {e}. Run `python -m src.data_loader` then `python -m src.url_model train`.")
    st.stop()

url = st.text_input("Website URL", placeholder="https://example.com")
fetch = st.checkbox("Download the page HTML for a more reliable check (recommended)", value=True)

if st.button("Check website", type="primary"):
    try:
        with st.spinner("Analysing..."):
            res = predict_url(url, fetch)
    except (ValueError, FileNotFoundError) as e:
        st.warning(str(e))
    else:
        p = res["fake_probability"]
        show = st.error if p >= 0.70 else (st.warning if p >= 0.30 else st.success)
        show(f"{res['prediction']}  (fake probability {p:.1%})")
        if not res["reliable"]:
            st.warning("Some indicators could not be measured, so treat this result as low confidence.")
        for n in res["notes"]:
            st.info(n)
        st.subheader("Indicators measured")
        st.caption("-1 phishing-like, 0 suspicious, 1 legitimate-like; blank = could not be measured")
        st.dataframe(pd.DataFrame({"indicator": res["features"].keys(), "value": res["features"].values()}),
                     hide_index=True)
st.divider()
st.caption("Limits: this reproduces a 2015-era research recipe with 21 of its 30 indicators, so it can be wrong "
           "in both directions (popular sites flagged, new phishing missed). Use it as a screening aid, not proof.")