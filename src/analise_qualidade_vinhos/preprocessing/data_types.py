"""
Módulo para padronização de tipos de dados
Funções para conversão e detecção automática de tipos
"""

import pandas as pd
from typing import Dict, Optional


def padroniza_tipos_dados(df, type_mapping=None, auto_detect=True, verbose=True):
    """
    Padroniza tipos de dados
    
    Args:
        df: DataFrame
        type_mapping: dict {'coluna': 'tipo'}
        auto_detect: tentar detectar tipos automaticamente
        verbose: mostrar alterações
    
    Returns:
        DataFrame com tipos padronizados
    """
    df_typed = df.copy()
    
    if auto_detect:
        # Auto-detectar alguns padrões
        for col in df_typed.columns:
            # Tentar converter para numérico se possível
            if df_typed[col].dtype == 'object':
                try:
                    # Testar conversão numérica
                    pd.to_numeric(df_typed[col], errors='raise')
                    df_typed[col] = pd.to_numeric(df_typed[col])
                except:
                    # Testar conversão para datetime
                    try:
                        pd.to_datetime(df_typed[col], errors='raise')
                        df_typed[col] = pd.to_datetime(df_typed[col])
                    except:
                        pass
    
    # Aplicar mapeamento específico
    if type_mapping:
        for col, dtype in type_mapping.items():
            if col in df_typed.columns:
                try:
                    if dtype == 'datetime':
                        df_typed[col] = pd.to_datetime(df_typed[col])
                    elif dtype == 'category':
                        df_typed[col] = df_typed[col].astype('category')
                    else:
                        df_typed[col] = df_typed[col].astype(dtype)
                except Exception as e:
                    print(f"⚠️  Erro ao converter {col} para {dtype}: {e}")
    
    if verbose:
        print("🔧 TIPOS DE DADOS:")
        print(df_typed.dtypes)
    
    return df_typed