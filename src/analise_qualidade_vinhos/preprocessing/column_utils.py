"""
Módulo para manipulação de colunas
Funções para limpeza, renomeação e padronização de nomes de colunas
"""

import pandas as pd
import re
from typing import Dict, Optional


def clean_columns(df, remove_spaces=True, lowercase=True, remove_special_chars=True, 
                    custom_replacements=None, verbose=True):
    """
    Limpa nomes das colunas
    
    Args:
        df: DataFrame
        remove_spaces: remover espaços
        lowercase: converter para minúsculas
        remove_special_chars: remover caracteres especiais
        custom_replacements: dict com substituições customizadas
        verbose: mostrar alterações
    
    Returns:
        DataFrame com colunas limpas
    """
    df_clean = df.copy()
    old_columns = df_clean.columns.tolist()
    new_columns = old_columns.copy()
    
    for i, col in enumerate(new_columns):
        # Aplicar transformações
        if lowercase:
            col = col.lower()
        
        if remove_spaces:
            col = col.replace(' ', '_')
        
        if remove_special_chars:
            col = re.sub(r'[^a-zA-Z0-9_]', '', col) # Regex
        
        # Substituições customizadas
        if custom_replacements:
            for old, new in custom_replacements.items():
                col = col.replace(old, new)
        
        new_columns[i] = col
    
    # Aplicar novos nomes
    df_clean.columns = new_columns
    
    if verbose and old_columns != new_columns:
        print("🏷️  COLUNAS RENOMEADAS:")
        for old, new in zip(old_columns, new_columns):
            if old != new:
                print(f"   '{old}' → '{new}'")
    
    return df_clean


def rename_columns(df, column_mapping, verbose=True):
    """
    Renomeia colunas específicas
    
    Args:
        df: DataFrame
        column_mapping: dict {'nome_antigo': 'nome_novo'}
        verbose: mostrar alterações
    
    Returns:
        DataFrame com colunas renomeadas
    """
    df_renamed = df.copy()
    
    # Verificar se colunas existem
    missing_cols = [col for col in column_mapping.keys() if col not in df_renamed.columns]
    if missing_cols:
        print(f"⚠️  Colunas não encontradas: {missing_cols}")
        column_mapping = {k: v for k, v in column_mapping.items() if k not in missing_cols}
    
    df_renamed = df_renamed.rename(columns=column_mapping)
    
    if verbose and column_mapping:
        print("🏷️  COLUNAS RENOMEADAS:")
        for old, new in column_mapping.items():
            print(f"   '{old}' → '{new}'")
    
    return df_renamed