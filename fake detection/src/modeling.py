"""Pipelines: every transformation lives INSIDE the pipeline, so it is fitted on training data only."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED


def build_baseline() -> Pipeline:
    return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])


def build_logreg() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])


def build_random_forest() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1)),
    ])


# Tuned with cross-validation on TRAINING data only.
PARAM_GRIDS = {
    "logistic_regression": {"clf__C": [0.01, 0.1, 1, 10]},
    "random_forest": {"clf__n_estimators": [100, 300],
                      "clf__max_depth": [None, 10, 20],
                      "clf__min_samples_leaf": [1, 3]},
}
BUILDERS = {"logistic_regression": build_logreg, "random_forest": build_random_forest}