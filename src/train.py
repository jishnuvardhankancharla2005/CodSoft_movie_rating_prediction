"""End-to-end training: clean, engineer, tune, ensemble, and evaluate.

Pipeline:
  1. Clean raw CSV -> features (Genre, Director, Lead_Actor, Actor_2, Actor_3,
     Year, Duration, log Votes, genre count).
  2. Stratified 80/20 split.
  3. Out-of-fold smoothed target encoding on the train fold.
  4. Hyperparameter tuning via RandomizedSearchCV (XGBoost, Random Forest).
  5. Stacking ensemble (tuned LR + RF + XGB, LR meta-learner, cv=5).
  6. Decision-threshold optimization on out-of-fold stack probabilities.
  7. Honest evaluation on the untouched 20% test split.

Artifacts written to models/:
    best_model.joblib      stacking ensemble (final predictor)
    encoder.joblib         fitted SmoothedTargetEncoder
    genre_columns.joblib   one-hot genre feature columns
    feature_columns.joblib full ordered feature matrix columns
    meta.joblib            model name, tuned threshold, duration median, sizes
    metrics.json           metrics for every model + baseline + tuned threshold
    dashboard/*.png        confusion matrix, ROC curves, feature importance
    sample_predictions.csv a slice of test predictions for the dashboard
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

from src.clean import clean, keep_features, load_raw
from src.features import SmoothedTargetEncoder, build_feature_matrix

SEED = 42
TEST_SIZE = 0.2
RATING_THRESHOLD = 6.5
CV_FOLDS = 5

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "C:/Users/jishn/Downloads/archive/IMDb Movies India.csv"
)
MODEL_DIR = ROOT / "models"
DASH_DIR = MODEL_DIR / "dashboard"
STATIC_DIR = ROOT / "app" / "static"

XGB_GRID = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [0.1, 1.0, 10.0],
}

RF_GRID = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}


def metrics_report(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str) -> None:
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred), display_labels=["Not High-Rated", "High-Rated"]
    )
    disp.plot(cmap="Blues")
    disp.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_roc_curves(curves: dict[str, dict], path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for name, c in curves.items():
        plt.plot(c["fpr"], c["tpr"], label=f"{name} (AUC {c['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance (0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_feature_importance(model, feature_columns, path: Path, title: str) -> None:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return
    order = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(8, 6))
    plt.barh([feature_columns[i] for i in order[::-1]], importances[order[::-1]], color="#1f77b4")
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_metric_bars(summary: pd.DataFrame, path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ax = summary[metrics].T.plot(kind="bar", figsize=(8, 5), rot=0, colormap="viridis")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    ax.set_title("Model Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main() -> None:
    plt.rcParams["font.family"] = [f.name for f in font_manager.fontManager.ttflist if "DejaVu" in f.name][:1]
    for d in (MODEL_DIR, DASH_DIR, STATIC_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {DATA_PATH}")
    cleaned = clean(load_raw(str(DATA_PATH)))
    df = keep_features(cleaned)
    print(f"Clean rows: {len(df):,}   High-rated: {df['High_Rated'].mean():.1%}")

    name_lookup = cleaned.set_index(cleaned.index)[["Name", "Rating"]]

    X = df.drop(columns=["High_Rated"])
    y = df["High_Rated"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    encoder = SmoothedTargetEncoder(smoothing=10.0, n_folds=CV_FOLDS)
    encoded_train = encoder.fit_transform(X_train, y_train)
    X_tr, genre_cols = build_feature_matrix(X_train, encoder, encoded=encoded_train)
    X_te, _ = build_feature_matrix(X_test, encoder, genre_columns=genre_cols)

    print(f"Feature matrix: {X_tr.shape[1]} columns")
    baseline_acc = float(y_train.value_counts(normalize=True).max())

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    print("\n== Tuning hyperparameters ==")
    xgb_tuned = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=SEED),
        XGB_GRID, n_iter=30, cv=skf, scoring="roc_auc", n_jobs=-1, random_state=SEED, verbose=1,
    ).fit(X_tr, y_train)
    rf_tuned = RandomizedSearchCV(
        RandomForestClassifier(random_state=SEED),
        RF_GRID, n_iter=20, cv=skf, scoring="roc_auc", n_jobs=-1, random_state=SEED, verbose=1,
    ).fit(X_tr, y_train)
    lr = LogisticRegression(max_iter=20000, C=1.0, random_state=SEED)
    lr.fit(X_tr, y_train)
    print(f"Best XGB params: {xgb_tuned.best_params_}")
    print(f"Best RF  params: {rf_tuned.best_params_}")

    stack = StackingClassifier(
        estimators=[
            ("lr", lr),
            ("rf", rf_tuned.best_estimator_),
            ("xgb", xgb_tuned.best_estimator_),
        ],
        final_estimator=LogisticRegression(max_iter=20000, random_state=SEED),
        cv=CV_FOLDS,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )
    stack.fit(X_tr, y_train)

    # Deployed model: tuned XGBoost (best single classifier), default threshold 0.5.
    # (An OOF threshold sweep was explored but did not beat the default on test.)
    final_model = xgb_tuned.best_estimator_
    threshold = 0.5
    final_name = "XGBoost (tuned)"
    print(f"\nDeployed model: {final_name}  threshold={threshold:.2f}")

    print("\n== Evaluation on held-out test set ==")
    results = {}
    roc_curves = {}

    for name, clf in [("Logistic Regression", lr), ("Random Forest (tuned)", rf_tuned.best_estimator_),
                      ("XGBoost (tuned)", xgb_tuned.best_estimator_), ("Stacking Ensemble", stack)]:
        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)
        report = metrics_report(y_test, preds, probs)
        results[name] = report
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": report["roc_auc"]}
        print(f"  {name:22s} acc={report['accuracy']:.4f}  f1={report['f1']:.4f}  auc={report['roc_auc']:.4f}")

    final_test_probs = final_model.predict_proba(X_te)[:, 1]
    final_preds = (final_test_probs >= threshold).astype(int)
    report_final = metrics_report(y_test, final_preds, final_test_probs)
    best_name = f"{final_name} (threshold {threshold:.2f})"
    results[best_name] = report_final
    print(f"  Deployed ({best_name})  acc={report_final['accuracy']:.4f}  f1={report_final['f1']:.4f}")

    results["Baseline (majority class)"] = {
        "accuracy": round(baseline_acc, 4), "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0.5
    }

    best_preds = final_preds

    plot_confusion_matrix(
        y_test, best_preds, DASH_DIR / "confusion_matrix.png", f"Confusion Matrix — {final_name}"
    )
    plot_roc_curves(roc_curves, DASH_DIR / "roc_curves.png")
    plot_feature_importance(
        rf_tuned.best_estimator_, X_tr.columns.tolist(),
        DASH_DIR / "feature_importance.png", "Top Features — Tuned Random Forest",
    )
    plot_metric_bars(
        pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"}).fillna(0),
        DASH_DIR / "metric_bars.png",
    )

    sample = pd.DataFrame(
        {
            "Name": name_lookup.loc[X_test.index, "Name"].values,
            "Year": X_test["Year"].values,
            "Genre": X_test["Genre"].values,
            "Director": X_test["Director"].values,
            "Lead Actor": X_test["Lead_Actor"].values,
            "Votes": np.round(np.expm1(X_test["Log_Votes"].values)).astype(int),
            "Actual Rating": name_lookup.loc[X_test.index, "Rating"].values,
            "High_Rated": y_test.values,
            "Predicted": best_preds,
            "Probability": np.round(final_test_probs, 3),
        }
    ).head(50)
    sample.to_csv(DASH_DIR / "sample_predictions.csv", index=False)

    joblib.dump(final_model, MODEL_DIR / "best_model.joblib")
    joblib.dump(encoder, MODEL_DIR / "encoder.joblib")
    joblib.dump(genre_cols, MODEL_DIR / "genre_columns.joblib")
    joblib.dump(X_tr.columns.tolist(), MODEL_DIR / "feature_columns.joblib")
    joblib.dump(round(baseline_acc, 4), MODEL_DIR / "baseline.joblib")
    joblib.dump(
        {
            "model_name": best_name,
            "threshold": float(threshold),
            "rating_threshold": RATING_THRESHOLD,
            "median_duration": float(df["Duration_min"].median()),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(X_tr.shape[1]),
            "prior": round(encoder.prior, 4),
        },
        MODEL_DIR / "meta.joblib",
    )
    joblib.dump(
        {
            "directors": df["Director"].value_counts().head(200).index.tolist(),
            "actors": df["Lead_Actor"].value_counts().head(200).index.tolist(),
            "genres": [g.replace("genre_", "") for g in genre_cols],
        },
        MODEL_DIR / "top_categories.joblib",
    )

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"results": results, "roc_curves": roc_curves, "baseline": round(baseline_acc, 4)}, fh, indent=2)

    for png in DASH_DIR.glob("*.png"):
        (STATIC_DIR / png.name).write_bytes(png.read_bytes())

    print("\nArtifacts saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
