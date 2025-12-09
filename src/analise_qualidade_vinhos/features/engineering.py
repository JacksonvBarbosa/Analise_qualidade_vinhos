"""
Feature engineering utilities used across training and inference.
The functions here are intentionally simple to match a junior-friendly stack.
"""

from __future__ import annotations

import pandas as pd

QUALITY_THRESHOLD = 6  # >= 6 = Alta qualidade, < 6 = Baixa qualidade

TARGET_LABELS = {
    "low": "Baixa qualidade",
    "high": "Alta qualidade",
}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case for easier maintenance."""
    column_map = {
        "fixed acidity": "fixed_acidity",
        "volatile acidity": "volatile_acidity",
        "citric acid": "citric_acid",
        "residual sugar": "residual_sugar",
        "chlorides": "chlorides",
        "free sulfur dioxide": "free_sulfur_dioxide",
        "total sulfur dioxide": "total_sulfur_dioxide",
        "density": "density",
        "pH": "ph",
        "sulphates": "sulphates",
        "alcohol": "alcohol",
        "quality": "quality",
    }
    existing = {k: v for k, v in column_map.items() if k in df.columns}
    return df.rename(columns=existing)


def bucket_quality(quality: float) -> str:
    """Convert numeric quality into binary classification: Alta qualidade (>=6) or Baixa qualidade (<6)."""
    if quality >= QUALITY_THRESHOLD:
        return TARGET_LABELS["high"]
    return TARGET_LABELS["low"]


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features that help tree-based models capture wine quality patterns."""
    df = df.copy()
    
    # Ratios importantes para qualidade do vinho
    df["density_alcohol_ratio"] = df["density"] / (df["alcohol"] + 1e-6)
    df["sulphates_alcohol_ratio"] = df["sulphates"] / (df["alcohol"] + 1e-6)
    df["total_free_sulfur_ratio"] = df["total_sulfur_dioxide"] / (
        df["free_sulfur_dioxide"] + 1e-6
    )
    
    # Índices compostos
    df["acidity_index"] = (
        df["fixed_acidity"] + df["volatile_acidity"] + df["citric_acid"]
    )
    df["total_acidity"] = df["fixed_acidity"] + df["volatile_acidity"]
    
    # Interações importantes
    df["sugar_sulphates_interaction"] = df["residual_sugar"] * df["sulphates"]
    df["alcohol_sulphates"] = df["alcohol"] * df["sulphates"]
    df["ph_acidity_interaction"] = df["ph"] * df["acidity_index"]
    
    # Features polinomiais (grau 2) para variáveis mais importantes
    df["alcohol_squared"] = df["alcohol"] ** 2
    df["volatile_acidity_squared"] = df["volatile_acidity"] ** 2
    df["sulphates_squared"] = df["sulphates"] ** 2
    
    # Razões de enxofre (importante para qualidade)
    df["sulfur_efficiency"] = df["free_sulfur_dioxide"] / (df["total_sulfur_dioxide"] + 1e-6)
    
    # Balance de acidez (relação entre tipos de acidez)
    df["citric_fixed_ratio"] = df["citric_acid"] / (df["fixed_acidity"] + 1e-6)
    df["volatile_fixed_ratio"] = df["volatile_acidity"] / (df["fixed_acidity"] + 1e-6)
    
    # Densidade ajustada (relação com açúcar residual)
    df["density_sugar_interaction"] = df["density"] * df["residual_sugar"]
    
    # Substituir infinitos e valores muito grandes por NaN
    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    
    # Preencher NaN com valores medianos das colunas (evita perda de dados)
    for col in df.select_dtypes(include=['float64', 'float32']).columns:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    return df


def build_feature_matrix(raw_df: pd.DataFrame, add_quality_label: bool = True) -> pd.DataFrame:
    """
    Create the modeling table:
    - clean column names
    - drop duplicates
    - engineer new features
    - add categorical quality bucket (target)
    """
    df = rename_columns(raw_df)
    df = df.drop_duplicates()
    df = create_interaction_features(df)

    if add_quality_label:
        if "quality" not in df.columns:
            raise ValueError("Coluna 'quality' não encontrada no dataset bruto.")
        df["quality_label"] = df["quality"].apply(bucket_quality)

    return df