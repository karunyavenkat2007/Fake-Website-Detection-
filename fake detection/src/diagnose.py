"""Explain WHY a URL got its score: compare each measured indicator with the real dataset.

    python -m src.diagnose https://www.python.org

For every indicator it prints the value measured from the URL and how common that value is among
LEGITIMATE vs FAKE sites in the training data. A value that is rare for legit sites but common for
fake ones is what pushes the prediction toward FAKE.
"""
import argparse

import numpy as np
import pandas as pd

from .config import RAW_TARGET
from .data_loader import load_dataset
from .url_model import URL_FEATURES, extract_features, predict_url


def diagnose(url: str, fetch: bool = True) -> pd.DataFrame:
    df = load_dataset().drop_duplicates()
    legit, fake = df[df[RAW_TARGET] == 1], df[df[RAW_TARGET] == -1]
    feats, notes = extract_features(url, fetch)
    rows = []
    for f in URL_FEATURES:
        v = feats.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            rows.append({"indicator": f, "measured": "n/a", "legit_share": None, "fake_share": None,
                         "fake/legit": None})
            continue
        pl, pf = float((legit[f] == v).mean()), float((fake[f] == v).mean())
        rows.append({"indicator": f, "measured": int(v), "legit_share": round(pl, 3),
                     "fake_share": round(pf, 3), "fake/legit": round(pf / max(pl, 1e-3), 2)})
    return pd.DataFrame(rows).sort_values("fake/legit", ascending=False, na_position="last"), notes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    res = predict_url(a.url, not a.no_fetch)
    table, notes = diagnose(a.url, not a.no_fetch)
    print(f"\n{res['url']}  ->  {res['prediction']}  (fake probability {res['fake_probability']:.1%})\n")
    print(table.to_string(index=False))
    print("\nRead it like this: 'fake/legit' above ~2 means that value is much more typical of fake "
          "sites than legitimate ones in the dataset. The top rows are what pushes the score up.")
    for n in notes:
        print("NOTE:", n)