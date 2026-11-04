"""Central configuration: paths, seed, schema of the dataset."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
SAMPLES_DIR = ROOT / "sample_inputs"

DATA_PATH = DATA_DIR / "phishing_websites.csv"
MODEL_PATH = MODELS_DIR / "fake_website_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Dataset: UCI "Phishing Websites" (Mohammad, Thabtah & McCluskey), CC BY 4.0.
UCI_URL = "https://archive.ics.uci.edu/static/public/327/phishing+websites.zip"
RAW_TARGET = "Result"          # -1 = phishing, 1 = legitimate (original coding)
TARGET = "is_fake"             # 1 = fake/phishing website, 0 = legitimate
MIN_RECORDS = 500
MIN_PREDICTORS = 6

FEATURES = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain",
    "SSLfinal_State", "Domain_registeration_length", "Favicon", "port",
    "HTTPS_token", "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
    "Submitting_to_email", "Abnormal_URL", "Redirect", "on_mouseover",
    "RightClick", "popUpWidnow", "Iframe", "age_of_domain", "DNSRecord",
    "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page",
    "Statistical_report",
]

# Every predictor is pre-encoded: -1 = phishing-like, 0 = suspicious, 1 = legitimate-like.
ALLOWED_VALUES = (-1, 0, 1)

# Columns used for the subgroup (error-slice) diagnostics.
SLICE_FEATURES = ["SSLfinal_State", "URL_Length", "having_Sub_Domain", "web_traffic"]