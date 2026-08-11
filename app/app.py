"""Flask web app: interactive 3D prediction UI + results dashboard.

Endpoints:
    /            -> prediction form
    /predict     -> POST, returns verdict + confidence
    /dashboard   -> training results dashboard
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, render_template, request

from src.features import SmoothedTargetEncoder, build_feature_matrix

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
POSTER_DIR = ROOT / "app" / "static" / "posters"

app = Flask(__name__)

model = joblib.load(MODEL_DIR / "best_model.joblib")
encoder: SmoothedTargetEncoder = joblib.load(MODEL_DIR / "encoder.joblib")
genre_cols = joblib.load(MODEL_DIR / "genre_columns.joblib")
feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")
meta = joblib.load(MODEL_DIR / "meta.joblib")
baseline = joblib.load(MODEL_DIR / "baseline.joblib")
top = joblib.load(MODEL_DIR / "top_categories.joblib")

with open(MODEL_DIR / "metrics.json", encoding="utf-8") as fh:
    metrics = json.load(fh)

THRESHOLD = float(meta["threshold"])
MEDIAN_DURATION = float(meta["median_duration"])

FEATURE_LABELS = {
    "Year": "Release year",
    "Duration_min": "Runtime (min)",
    "Log_Votes": "Vote buzz",
    "Genre_Count": "Genre count",
    "Director_enc": "Director track record",
    "Lead_Actor_enc": "Lead actor track record",
    "Actor_2_enc": "Supporting cast (2nd)",
    "Actor_3_enc": "Supporting cast (3rd)",
}


def explain(matrix: pd.DataFrame) -> list[dict]:
    """SHAP feature contributions for the single-row feature matrix."""
    booster = model.get_booster()
    contribs = booster.predict(xgb.DMatrix(matrix), pred_contribs=True)[0]
    items = []
    for name, value in zip(feature_cols, contribs[:-1]):
        if name.startswith("genre_"):
            label = f"Genre: {name.replace('genre_', '')}"
        else:
            label = FEATURE_LABELS.get(name, name)
        items.append({"feature": label, "impact": round(float(value), 4)})
    items.sort(key=lambda x: -abs(x["impact"]))
    return items[:8]


def build_input_row(
    genre: str,
    director: str,
    actor: str,
    year: int,
    votes: int,
    duration: float | None = None,
    actor2: str | None = None,
    actor3: str | None = None,
    genre_count: int = 1,
) -> pd.DataFrame:
    row = pd.DataFrame(
        {
            "Genre": [genre],
            "Director": [director],
            "Lead_Actor": [actor],
            "Actor_2": [actor2 or "Unknown"],
            "Actor_3": [actor3 or "Unknown"],
            "Year": [float(year)],
            "Duration_min": [float(duration) if duration else MEDIAN_DURATION],
            "Log_Votes": [float(math.log1p(max(votes, 1)))],
            "Genre_Count": [float(genre_count)],
        }
    )
    encoded = encoder.transform(row)
    matrix, _ = build_feature_matrix(row, encoder, genre_columns=genre_cols, encoded=encoded)
    return matrix[feature_cols]


@app.route("/")
def index():
    posters = sorted(p.name for p in POSTER_DIR.glob("*.jpg"))
    return render_template(
        "index.html",
        genres=top["genres"],
        directors=top["directors"],
        actors=top["actors"],
        model_name=meta["model_name"],
        posters=posters,
        baseline=baseline,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        genre = request.form["genre"]
        director = request.form["director"].strip() or "Unknown"
        actor = request.form["actor"].strip() or "Unknown"
        year = int(request.form["year"])
        votes = int(request.form["votes"])
    except (KeyError, ValueError):
        return jsonify({"error": "Please fill all required fields with valid numbers."}), 400

    if not 1900 <= year <= 2026:
        return jsonify({"error": "Year must be between 1900 and 2026."}), 400
    if votes < 0:
        return jsonify({"error": "Votes cannot be negative."}), 400

    duration_raw = request.form.get("duration", "").strip()
    actor2 = request.form.get("actor2", "").strip()
    actor3 = request.form.get("actor3", "").strip()

    duration = None
    if duration_raw:
        try:
            duration = int(duration_raw)
        except ValueError:
            return jsonify({"error": "Duration must be a number."}), 400
        if not 10 <= duration <= 600:
            return jsonify({"error": "Duration must be between 10 and 600 minutes."}), 400

    matrix = build_input_row(
        genre, director, actor, year, votes, duration=duration, actor2=actor2, actor3=actor3
    )
    prob = float(model.predict_proba(matrix)[0, 1])
    verdict = "High-Rated" if prob >= THRESHOLD else "Not High-Rated"

    return jsonify(
        {
            "verdict": verdict,
            "probability": round(prob, 4),
            "confidence": round(max(prob, 1 - prob) * 100, 1),
            "model": meta["model_name"],
            "threshold": THRESHOLD,
            "rating_threshold": meta["rating_threshold"],
            "baseline": baseline,
            "contributions": explain(matrix),
        }
    )


@app.route("/dashboard")
def dashboard():
    table = pd.read_csv(MODEL_DIR / "dashboard" / "sample_predictions.csv")
    return render_template(
        "dashboard.html",
        metrics=metrics,
        meta=meta,
        baseline=baseline,
        sample=table.to_dict(orient="records"),
        columns=list(table.columns),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
