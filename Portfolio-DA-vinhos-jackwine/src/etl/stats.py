# src/etl/stats.py

import pandas as pd
import numpy as np

def calcula_skew_kurtosis(df: pd.DataFrame):
    """Retorna o skewness e kurtosis de cada coluna numérica do dataframe."""
    skewness = df.skew(numeric_only=True)
    kurtosis = df.kurtosis(numeric_only=True)
    return skewness, kurtosis

def obter_dados_nulos(df: pd.DataFrame):
    """Retorna a quantidade e a porcentagem de valores nulos em cada coluna."""
    total_missing = df.isnull().sum()
    percent_missing = (total_missing / len(df)) * 100
    return pd.DataFrame({
        'Valores Nulos': total_missing,
        '% Nulos': percent_missing
    }).sort_values(by='Valores Nulos', ascending=False)

def detecta_outliers_iqr(df: pd.DataFrame, column: str):
    """Retorna os outliers da coluna especificada com base no método IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def matrix_correlacao(df: pd.DataFrame):
    """Retorna a matriz de correlação entre colunas numéricas."""
    return df.corr(numeric_only=True)
