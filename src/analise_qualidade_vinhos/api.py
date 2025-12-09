"""FastAPI app to expose the trained model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from analise_qualidade_vinhos.config.settings import MODEL_DIR
from analise_qualidade_vinhos.pipeline.predict import load_model, predict_from_dataframe
from analise_qualidade_vinhos.pipeline.train import train_model

app = FastAPI(
    title="Wine Quality Service",
    description="API simples para pontuar qualidade de vinhos (2 faixas).",
    version="0.1.0",
)


class WineSample(BaseModel):
    fixed_acidity: float = Field(..., ge=0)
    volatile_acidity: float = Field(..., ge=0)
    citric_acid: float = Field(..., ge=0)
    residual_sugar: float = Field(..., ge=0)
    chlorides: float = Field(..., ge=0)
    free_sulfur_dioxide: float = Field(..., ge=0)
    total_sulfur_dioxide: float = Field(..., ge=0)
    density: float = Field(..., ge=0)
    ph: float = Field(..., ge=0)
    sulphates: float = Field(..., ge=0)
    alcohol: float = Field(..., ge=0)


@lru_cache(maxsize=1)
def get_or_train_model():
    model_path = MODEL_DIR / "wine_quality_model.joblib"
    if not model_path.exists():
        train_model(model_path=model_path)
    return load_model(model_path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(samples: List[WineSample]) -> dict:
    if not samples:
        raise HTTPException(status_code=400, detail="Envie pelo menos uma amostra.")
    df = pd.DataFrame([s.model_dump() for s in samples])
    model = get_or_train_model()
    preds = predict_from_dataframe(model, df)
    return {"predictions": preds}


