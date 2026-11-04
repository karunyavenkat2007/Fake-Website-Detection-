"""Train baseline + 2 algorithms, evaluate once on an untouched holdout, save the model.

Run:  python -m src.train
"""
import argparse
import json
import platform
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .audit import run_audit
from .config import (CV_FOLDS, DATA_PATH, FEATURES, METADATA_PATH, MODEL_PATH, MODELS_DIR,
                     RAW_TARGET, REPORTS_DIR, SAMPLES_DIR, SEED, SLICE_FEATURES, TARGET,
                     TEST_SIZE)
from .data_loader import load_dataset
from .modeling import BUILDERS, PARAM_GRIDS, build_baseline
from .utils import file_sha256, get_logger

log = get_logger()


def metrics_row(name, model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else pred
    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "precision_fake": precision_score(y, pred, zero_division=0),
        "recall_fake": recall_score(y, pred, zero_division=0),
        "f1_fake": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(proba)) > 1 else 0.5,
    }


def slice_report(model, X, y, feature):
    """Per-subgroup diagnostics: how does the model behave for each value of `feature`?"""
    rows = []
    pred = pd.Series(model.predict(X), index=X.index)
    for value, idx in X.groupby(feature).groups.items():
        yt, yp = y.loc[idx], pred.loc[idx]
        rows.append({"slice_feature": feature, "value": int(value), "n": len(idx),
                     "fake_share": round(float(yt.mean()), 3),
                     "accuracy": round(accuracy_score(yt, yp), 4),
                     "recall_fake": round(recall_score(yt, yp, zero_division=0), 4),
                     "false_positive_rate": round(float(((yp == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1)), 4)})
    return pd.DataFrame(rows)


def main(data_path=DATA_PATH):
    for d in (MODELS_DIR, REPORTS_DIR, SAMPLES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1) load + validate + audit (duplicates removed BEFORE the split)
    raw = load_dataset(data_path)
    df = run_audit(raw)
    log.info("Rows before/after de-duplication: %d -> %d", len(raw), len(df))
    X = df[FEATURES].astype(float)
    y = (df[RAW_TARGET] == -1).astype(int).rename(TARGET)   # 1 = fake website

    # 2) split first; nothing is fitted on the test set
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    log.info("Train=%d  Test=%d  (stratified, seed=%d)", len(X_tr), len(X_te), SEED)

    # 3) baseline + tuned models (cross-validation on training folds only)
    baseline = build_baseline().fit(X_tr, y_tr)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    tuned, tuning_logs = {}, []
    for name, builder in BUILDERS.items():
        gs = GridSearchCV(builder(), PARAM_GRIDS[name], cv=cv, scoring="f1", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        tuned[name] = gs
        res = pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
        res.insert(0, "model", name)
        tuning_logs.append(res)
        log.info("%s best CV F1=%.4f params=%s", name, gs.best_score_, gs.best_params_)
    pd.concat(tuning_logs).to_csv(REPORTS_DIR / "tuning_log.csv", index=False)

    # 4) model selection uses CV score on TRAINING data, not the test set
    best_name = max(tuned, key=lambda n: tuned[n].best_score_)
    best_model = tuned[best_name].best_estimator_
    log.info("Selected model: %s (by training-fold CV F1)", best_name)

    # 5) single evaluation on the same untouched holdout for all three
    models = {"baseline_majority": baseline, **{n: g.best_estimator_ for n, g in tuned.items()}}
    comp = pd.DataFrame([metrics_row(n, m, X_te, y_te) for n, m in models.items()]).round(4)
    comp.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    log.info("\n%s", comp.to_string(index=False))

    # FR-04: confusion matrices + class metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (n, m) in zip(axes, models.items()):
        ConfusionMatrixDisplay(confusion_matrix(y_te, m.predict(X_te)),
                               display_labels=["legit", "fake"]).plot(ax=ax, colorbar=False)
        ax.set_title(n)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    report = classification_report(y_te, best_model.predict(X_te), target_names=["legit", "fake"])
    (REPORTS_DIR / "classification_report_best.txt").write_text(report, encoding="utf-8")

    if best_name == "random_forest" or hasattr(best_model.named_steps["clf"], "feature_importances_"):
        imp = pd.Series(best_model.named_steps["clf"].feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(7, 8))
        imp.plot.barh(ax=ax)
        ax.set_title("Feature importance (random forest)")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)

    # FR-05: >= 2 error slices + worst mistakes for error analysis
    slices = pd.concat([slice_report(best_model, X_te, y_te, f) for f in SLICE_FEATURES])
    slices.to_csv(REPORTS_DIR / "error_slices.csv", index=False)
    proba = best_model.predict_proba(X_te)[:, 1]
    errors = X_te.assign(true_is_fake=y_te, predicted_fake_probability=proba.round(3))
    errors = errors[(proba >= 0.5).astype(int) != y_te]
    errors.assign(confidence_in_wrong_answer=np.abs(errors["predicted_fake_probability"] - errors["true_is_fake"]))\
        .sort_values("confidence_in_wrong_answer", ascending=False).head(20)\
        .to_csv(REPORTS_DIR / "worst_errors.csv")

    # FR-06: model artifact + metadata + sample inputs for mentor testing
    joblib.dump(best_model, MODEL_PATH)
    row = comp[comp.model == best_name].iloc[0].to_dict()
    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": best_name, "best_params": tuned[best_name].best_params_,
        "seed": SEED, "test_size": TEST_SIZE, "n_train": len(X_tr), "n_test": len(X_te),
        "dataset_file": str(data_path.name), "dataset_sha256": file_sha256(data_path),
        "dataset_source": "UCI Phishing Websites (id 327), CC BY 4.0",
        "target": f"{TARGET}: 1 = fake/phishing, 0 = legitimate", "features": FEATURES,
        "holdout_metrics": {k: v for k, v in row.items() if k != "model"},
        "python": platform.python_version(), "sklearn": sklearn.__version__,
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    one_fake = X_te[y_te == 1].iloc[0].astype(int).to_dict()
    one_legit = X_te[y_te == 0].iloc[0].astype(int).to_dict()
    (SAMPLES_DIR / "fake_example.json").write_text(json.dumps(one_fake, indent=2), encoding="utf-8")
    (SAMPLES_DIR / "legit_example.json").write_text(json.dumps(one_legit, indent=2), encoding="utf-8")
    log.info("Saved model to %s", MODEL_PATH)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_PATH))
    from pathlib import Path
    main(Path(ap.parse_args().data))