# libs
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional

# Remove nulos
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

# Limpa rotulo das colunas
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


# Renomear colunas
def rename_columns(df, column_mapping, verbose=True):
    """
    Renomeia colunas específicas
    
    Args:
        df: DataFrame
        column_mapping: list {'nome_novo', 'nome_novo_1', ... 'nome_novo_n'}
        column_mapping: dict {'nome_antigo': 'nome_novo'}
        verbose: mostrar alterações
    
    Returns:
        DataFrame com colunas renomeadas
    """
    df_renamed = df.copy()
    
    # Verificar se colunas existem
    #colunas = {col: column_mapping[i] for i, col in enumerate(df.columns)}
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

# Pradonizar tipos de dados
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


# Remover Duplicatas
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