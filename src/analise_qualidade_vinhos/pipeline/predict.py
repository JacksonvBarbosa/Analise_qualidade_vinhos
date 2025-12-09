"""Small helpers to load the trained pipeline and score new samples."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from joblib import load

from analise_qualidade_vinhos.config.settings import MODEL_DIR
from analise_qualidade_vinhos.features.engineering import build_feature_matrix


def load_model(model_path: Path | None = None):
    if model_path is None:
        model_path = MODEL_DIR / "wine_quality_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}. Treine antes de prever.")
    return load(model_path)


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used at training time."""
    featured = build_feature_matrix(df, add_quality_label=False)
    for col in ["quality_label", "quality"]:
        if col in featured.columns:
            featured = featured.drop(columns=[col])
    return featured


def predict_from_dataframe(model, df: pd.DataFrame) -> List[str]:
    prepared = prepare_input(df)
    predictions = model.predict(prepared)
    return predictions.tolist()

