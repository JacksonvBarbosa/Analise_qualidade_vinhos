"""
Módulo para limpeza de dados
Funções para remoção de nulos e duplicatas
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


def remove_nulls(df, strategy='drop', columns=None, fill_value=None, verbose=True):
    """
    Remove ou trata valores nulos
    
    Args:
        df: DataFrame
        strategy: 'drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_value'
        columns: colunas específicas (None = todas)
        fill_value: valor para preencher quando strategy='fill_value'
        verbose: mostrar informações
    
    Returns:
        DataFrame tratado
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns
    
    nulls_before = df_clean[columns].isnull().sum().sum()
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    elif strategy == 'fill_mean':
        for col in columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    elif strategy == 'fill_median':
        for col in columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    elif strategy == 'fill_mode':
        for col in columns:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    elif strategy == 'fill_value':
        df_clean[columns] = df_clean[columns].fillna(fill_value)
    
    nulls_after = df_clean[columns].isnull().sum().sum()
    
    if verbose:
        print(f"🧹 Valores nulos: {nulls_before} → {nulls_after}")
        if strategy == 'drop':
            rows_removed = len(df) - len(df_clean)
            print(f"📉 Linhas removidas: {rows_removed} ({rows_removed/len(df)*100:.2f}%)")
    
    return df_clean


def remove_duplicates(df, columns=None, keep='first', verbose=True):
    """
    Remove duplicatas
    
    Args:
        df: DataFrame
        columns: colunas para considerar (None = todas)
        keep: 'first', 'last' ou False
        verbose: mostrar informações
    
    Returns:
        DataFrame sem duplicatas
    """
    df_clean = df.copy()
    
    duplicates_before = df_clean.duplicated(subset=columns).sum()
    df_clean = df_clean.drop_duplicates(subset=columns, keep=keep)
    duplicates_removed = duplicates_before
    
    if verbose:
        print(f"🔄 Duplicatas removidas: {duplicates_removed}")
        print(f"📊 Linhas: {len(df)} → {len(df_clean)}")
    
    return df_clean