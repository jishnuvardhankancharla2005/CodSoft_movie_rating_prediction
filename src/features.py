"""Feature engineering: out-of-fold smoothed target encoding + one-hot genre.

The categorical features Director, Lead_Actor, Actor_2 and Actor_3 have very
high cardinality, so one-hot encoding is impractical and loses signal for rare
categories. Instead we target-encode them:

    encoded(cat) = (count * mean + smoothing * prior) / (count + smoothing)

where prior is the global High_Rated rate. To prevent leakage we encode each
training fold using statistics learned on the *other* folds (out-of-fold), and
encode the held-out test set using statistics learned on the full training set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DEFAULT_CAT_COLS = ["Director", "Lead_Actor", "Actor_2", "Actor_3"]
NUMERIC_COLS = ["Year", "Duration_min", "Log_Votes", "Genre_Count"]


@dataclass
class SmoothedTargetEncoder:
    """Persistable smoothed target encoder fitted on training data."""

    smoothing: float = 10.0
    n_folds: int = 5
    cat_cols: list[str] = field(default_factory=lambda: DEFAULT_CAT_COLS)
    maps: dict[str, dict[str, float]] = field(default_factory=dict)
    prior: float = 0.0

    def _fit_single(self, series: pd.Series, y: pd.Series) -> dict[str, float]:
        stats = (
            pd.DataFrame({"cat": series, "y": y})
            .groupby("cat")
            .agg(cnt=("y", "size"), mean=("y", "mean"))
        )
        return {
            cat: (row["cnt"] * row["mean"] + self.smoothing * self.prior) / (row["cnt"] + self.smoothing)
            for cat, row in stats.iterrows()
        }

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Return a DataFrame with out-of-fold encoded columns for X."""
        self.prior = float(np.mean(y))
        encoded = pd.DataFrame(index=X.index)

        for col in self.cat_cols:
            enc_values = pd.Series(np.nan, index=X.index, dtype="float64")
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            for train_idx, val_idx in skf.split(X, y):
                train_map = self._fit_single(X.iloc[train_idx][col], y.iloc[train_idx])
                enc_values.iloc[val_idx] = X.iloc[val_idx][col].map(train_map)
            encoded[f"{col}_enc"] = enc_values.fillna(self.prior)

        self.maps = {col: self._fit_single(X[col], y) for col in self.cat_cols}
        return encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode X using the already-fitted maps; unseen categories -> prior."""
        encoded = pd.DataFrame(index=X.index)
        for col in self.cat_cols:
            encoded[f"{col}_enc"] = X[col].map(self.maps.get(col, {})).fillna(self.prior)
        return encoded


def genre_one_hot(series: pd.Series, columns: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode the (already single) primary genre."""
    dummies = pd.get_dummies(series, prefix="genre").astype(int)
    if columns is not None:
        missing = [c for c in columns if c not in dummies.columns]
        for col in missing:
            dummies[col] = 0
        dummies = dummies[columns]
    return dummies.reset_index(drop=True)


def numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature block: Year, Duration, log(Votes), genre count."""
    return X[NUMERIC_COLS].astype("float64").reset_index(drop=True)


def build_feature_matrix(
    X: pd.DataFrame,
    encoder: SmoothedTargetEncoder | None,
    genre_columns: list[str] | None = None,
    encoded: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the final feature matrix from genre one-hot + encoded cats + numerics.

    Fit-transform path: pass encoder=None, genre_columns=None, encoded=None and the
    encoder will be fitted out-of-fold while genre columns are derived and returned
    via the returned genre list. For predict-time reuse, call with the saved encoder
    and genre_columns.
    """
    cats = encoded if encoded is not None else encoder.transform(X)
    genre_oh = genre_one_hot(X["Genre"].reset_index(drop=True), columns=genre_columns)
    nums = numeric_frame(X)
    matrix = pd.concat([genre_oh, cats.reset_index(drop=True), nums], axis=1)
    return matrix, (genre_oh.columns.tolist() if genre_columns is None else genre_columns)
