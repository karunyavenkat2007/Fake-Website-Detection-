"""Check a REAL website: turn a URL into the dataset's indicators, then predict.

Only indicators that can be computed from the URL, a DNS lookup, the TLS certificate and the page's
HTML are used (21 of the 30). The remaining 9 (WHOIS age, traffic rank, PageRank, Google index...)
need paid/defunct services, so a second model is trained on just these 21 indicators.

    python -m src.url_model train
    python -m src.url_model check https://example.com

The page is downloaded as TEXT only and never executed or rendered.
"""
import argparse
import ipaddress
import json
import re
import socket
import ssl
import sys
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .config import CV_FOLDS, MODELS_DIR, RAW_TARGET, REPORTS_DIR, SEED, TARGET, TEST_SIZE
from .data_loader import load_dataset
from .utils import get_logger

log = get_logger()

URL_MODEL_PATH = MODELS_DIR / "url_model.joblib"
URL_METRICS_PATH = REPORTS_DIR / "url_model_metrics.json"

URL_FEATURES = [
    # from the URL text
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "port", "HTTPS_token",
    # from the network (TLS certificate, DNS)
    "SSLfinal_State", "DNSRecord",
    # from the downloaded HTML
    "Favicon", "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email",
    "Redirect", "on_mouseover", "RightClick", "Iframe",
]
PAGE_FEATURES = URL_FEATURES[11:]
NO_PAGE_FEATURES = URL_FEATURES[:11]          # used when the page cannot be downloaded
NO_PAGE_MODEL_PATH = MODELS_DIR / "url_model_no_page.joblib"

SHORTENERS = {"bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly",
              "bit.do", "cutt.ly", "rebrand.ly", "tiny.cc", "shorturl.at", "tr.im", "x.co"}
SECOND_LEVEL = {"co", "com", "org", "net", "gov", "edu", "ac"}


# ---------------------------------------------------------------- helpers
def normalize_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Please enter a website URL.")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw = "https://" + raw
    try:
        p = urlparse(raw)
        _ = p.port
    except ValueError:
        raise ValueError("That does not look like a valid URL.")
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("Only http:// and https:// URLs with a host name are supported.")
    return raw


def registered_domain(host: str) -> str:
    labels = (host or "").lower().removeprefix("www.").split(".")
    if len(labels) >= 3 and labels[-2] in SECOND_LEVEL and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return bool(re.fullmatch(r"0x[0-9a-fA-F]+|\d{8,10}", host or ""))


# ---------------------------------------------------------------- URL / network features
def lexical_features(url: str) -> dict:
    p = urlparse(url)
    host = (p.hostname or "").lower()
    labels = host.removeprefix("www.").split(".")
    dots = len(labels) - 1
    if len(labels) >= 3 and labels[-2] in SECOND_LEVEL and len(labels[-1]) == 2:
        dots -= 1
    n = len(url)
    return {
        "having_IP_Address": -1 if _is_ip(host) else 1,
        "URL_Length": 1 if n < 54 else (0 if n <= 75 else -1),
        "Shortining_Service": -1 if host.removeprefix("www.") in SHORTENERS else 1,
        "having_At_Symbol": -1 if "@" in url else 1,
        "double_slash_redirecting": -1 if url.rfind("//") > 7 else 1,
        "Prefix_Suffix": -1 if "-" in host else 1,
        "having_Sub_Domain": 1 if dots <= 1 else (0 if dots == 2 else -1),
        "port": 1 if p.port in (None, 80, 443) else -1,
        "HTTPS_token": -1 if "https" in host else 1,
    }


def ssl_state(url: str):
    p = urlparse(url)
    if p.scheme != "https":
        return -1
    try:
        with socket.create_connection((p.hostname, p.port or 443), timeout=6) as s:
            with ssl.create_default_context().wrap_socket(s, server_hostname=p.hostname):
                return 1
    except ssl.SSLCertVerificationError:
        return 0
    except (OSError, ssl.SSLError):
        return np.nan


def dns_record(host: str):
    try:
        socket.getaddrinfo(host, None)
        return 1
    except socket.gaierror:
        return -1
    except OSError:
        return np.nan


# ---------------------------------------------------------------- HTML features
class _Redirects(urllib.request.HTTPRedirectHandler):
    def __init__(self):
        self.count = 0

    def redirect_request(self, *args, **kwargs):
        self.count += 1
        return super().redirect_request(*args, **kwargs)


def fetch_html(url: str):
    """Download up to 1.5 MB of HTML as text. Returns (html, final_url, n_redirects)."""
    counter = _Redirects()
    opener = urllib.request.build_opener(counter)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (fake-website-detection student project)"})
    with opener.open(req, timeout=8) as r:
        raw = r.read(1_500_000)
        charset = r.headers.get_content_charset() or "utf-8"
        final = r.geturl()
    return raw.decode(charset, errors="replace"), final, counter.count


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.objects, self.anchors, self.tag_links = [], [], []
        self.forms, self.icons, self.iframe = [], [], False

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag in ("img", "audio", "embed", "video", "source") and a.get("src"):
            self.objects.append(a["src"].strip())
        elif tag == "a":
            self.anchors.append(a.get("href", "").strip())
        elif tag == "link":
            href = a.get("href", "").strip()
            if "icon" in a.get("rel", "").lower():
                self.icons.append(href)
            elif href:
                self.tag_links.append(href)
        elif tag == "script" and a.get("src"):
            self.tag_links.append(a["src"].strip())
        elif tag == "form":
            self.forms.append(a.get("action", "").strip())
        elif tag == "iframe":
            self.iframe = True


def _is_external(link: str, base_host: str) -> bool:
    link = link.strip()
    if link.startswith("//"):
        link = "http:" + link
    if not re.match(r"^https?://", link, re.I):
        return False                      # relative path, data:, mailto:, # ...
    host = urlparse(link).hostname or ""
    return registered_domain(host) != registered_domain(base_host)


def _pct(links, base_host, extra=lambda l: False) -> float:
    if not links:
        return 0.0
    return 100.0 * sum(1 for l in links if _is_external(l, base_host) or extra(l)) / len(links)


def _band(pct, low, high):
    return 1 if pct < low else (0 if pct <= high else -1)


def page_features(html: str, final_url: str, redirects: int) -> dict:
    base = urlparse(final_url).hostname or ""
    c = _Collector()
    try:
        c.feed(html)
    except Exception as e:                  # malformed HTML must not crash the checker
        log.warning("HTML parse problem: %s", e)
    unsafe_anchor = lambda l: l == "" or l.startswith("#") or l.lower().startswith("javascript")
    sfh = []
    for action in c.forms:
        sfh.append(-1 if action in ("", "about:blank") else (0 if _is_external(action, base) else 1))
    return {
        "Favicon": -1 if any(_is_external(i, base) for i in c.icons) else 1,
        "Request_URL": _band(_pct(c.objects, base), 22, 61),
        "URL_of_Anchor": _band(_pct(c.anchors, base, unsafe_anchor), 31, 67),
        "Links_in_tags": _band(_pct(c.tag_links, base), 17, 81),
        "SFH": min(sfh) if sfh else 1,
        "Submitting_to_email": -1 if (any(a.lower().startswith("mailto:") for a in c.forms)
                                      or re.search(r"\bmail\s*\(", html)) else 1,
        "Redirect": 1 if redirects <= 1 else (0 if redirects < 4 else -1),
        "on_mouseover": -1 if re.search(r"onmouseover\s*=\s*[\"'][^\"']*window\.status", html, re.I) else 1,
        "RightClick": -1 if re.search(r"event\.button\s*==\s*2|oncontextmenu\s*=\s*[\"']?\s*return\s+false",
                                      html, re.I) else 1,
        "Iframe": -1 if c.iframe else 1,
    }


def extract_features(url: str, fetch: bool = True):
    """Return (features dict over URL_FEATURES, list of human-readable notes)."""
    url = normalize_url(url)
    notes, feats = [], lexical_features(url)
    host = urlparse(url).hostname
    feats["SSLfinal_State"] = ssl_state(url)
    feats["DNSRecord"] = dns_record(host)
    if np.isnan(feats["SSLfinal_State"]):
        notes.append("Could not check the TLS certificate (network problem).")
    if feats["DNSRecord"] == -1:
        notes.append("The domain name does not resolve in DNS.")
    got_page = False
    if fetch:
        try:
            html, final, redirects = fetch_html(url)
            feats.update(page_features(html, final, redirects))
            got_page = True
        except Exception as e:
            notes.append(f"Could not download the page ({type(e).__name__}: {e}). "
                         "Page-based indicators were filled with typical values, so reliability is lower.")
            log.warning("fetch failed for %s: %s", url, e)
    else:
        notes.append("Page download skipped; page-based indicators were filled with typical values.")
    for f in PAGE_FEATURES:
        feats.setdefault(f, np.nan)
    feats["_page_analysed"] = got_page
    return feats, notes


# ---------------------------------------------------------------- model
def _train_one(features, model_path):
    """Train a random forest on the given indicators (same split/seed as the main model)."""
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    df = load_dataset().drop_duplicates().reset_index(drop=True)
    X = df[features].astype(float)
    y = (df[RAW_TARGET] == -1).astype(int).rename(TARGET)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))])
    grid = {"clf__n_estimators": [200], "clf__max_depth": [None, 15], "clf__min_samples_leaf": [1, 3]}
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED),
                      scoring="f1", n_jobs=-1).fit(X_tr, y_tr)
    pred = gs.predict(X_te)
    metrics = {"features": features, "best_params": gs.best_params_, "cv_f1": round(gs.best_score_, 4),
               "holdout_accuracy": round(accuracy_score(y_te, pred), 4),
               "holdout_precision_fake": round(precision_score(y_te, pred), 4),
               "holdout_recall_fake": round(recall_score(y_te, pred), 4),
               "holdout_f1_fake": round(f1_score(y_te, pred), 4)}
    joblib.dump(gs.best_estimator_, model_path)
    log.info("Saved %s: %s", model_path.name, metrics)
    return metrics


def train_url_model():
    """Train BOTH URL models: one with page indicators (21) and a fallback without them (11)."""
    out = {"with_page_html": _train_one(URL_FEATURES, URL_MODEL_PATH),
           "url_dns_ssl_only": _train_one(NO_PAGE_FEATURES, NO_PAGE_MODEL_PATH)}
    URL_METRICS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def verdict(p: float) -> str:
    """Three-way verdict: a forced yes/no at 50% hides how unsure the model is."""
    if p >= 0.70:
        return "FAKE / phishing"
    if p >= 0.30:
        return "SUSPICIOUS - not enough evidence"
    return "legitimate"


def predict_url(url: str, fetch: bool = True) -> dict:
    if not (URL_MODEL_PATH.exists() and NO_PAGE_MODEL_PATH.exists()):
        raise FileNotFoundError("URL models not found. Run `python -m src.url_model train` first.")
    feats, notes = extract_features(url, fetch)
    page_ok = feats.pop("_page_analysed")
    if page_ok:
        used, model_path = URL_FEATURES, URL_MODEL_PATH
    else:   # never guess missing page indicators: use the model trained without them
        used, model_path = NO_PAGE_FEATURES, NO_PAGE_MODEL_PATH
        notes.append("Page not analysed, so the URL/DNS/certificate-only model was used (less accurate).")
    row = pd.DataFrame([{f: feats[f] for f in used}], columns=used)
    proba = float(joblib.load(model_path).predict_proba(row)[0, 1])
    return {"url": normalize_url(url), "prediction": verdict(proba),
            "fake_probability": round(proba, 4), "page_analysed": page_ok,
            "model_used": "with page HTML" if page_ok else "URL + DNS + certificate only",
            "reliable": page_ok and not row.isna().any(axis=None),
            "features": {k: (None if pd.isna(v) else int(v)) for k, v in feats.items()}, "notes": notes}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    chk = sub.add_parser("check")
    chk.add_argument("url")
    chk.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    try:
        out = train_url_model() if args.cmd == "train" else predict_url(args.url, not args.no_fetch)
        print(json.dumps(out, indent=2))
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)