# libs
import pandas as pd
import numpy as np
from scipy.stats import zscore

def clean_columns(df):
    '''
        Substitui espaços e hifen das colunas, por underscore
    '''
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ","_")
        .str.replace("-","_")
    )
    return df

def remove_nulls(df):
    '''
        Remover nulos do dataframe
    '''
    return df.dropna()

def rename_colunns(df, new_list):
    '''
        Renomear colunas caso necessário.
        inserir o dataframe.
        Inserir a lista com os novos nomes.
    '''
    renomeia_colunas = {col: new_list[i] for i, col in enumerate(df.columns)}
    df_rename = df.rename(columns = renomeia_colunas)
    return df_rename

def remove_outliers(
    df: pd.DataFrame,
    colunas: list,
    metodo: str = "ambos",  # 'zscore', 'iqr' ou 'ambos'
    z_thres: float = 3.0,
    iqr_factor: float = 1.5,
    remover_zscore_colunas: bool = True
) -> pd.DataFrame:
    """
    Remove outliers usando Z-Score, IQR ou ambos.
    
    Parâmetros:
    - df: DataFrame original
    - colunas: lista de colunas numéricas para avaliar
    - metodo: 'zscore', 'iqr' ou 'ambos'
    - z_thres: limite do Z-Score (default=3.0)
    - iqr_factor: fator multiplicador do IQR (default=1.5)
    - remover_zscore_colunas: se True, remove colunas z_score_* após o filtro
    
    Retorna:
    - DataFrame sem os outliers
    """
    df_temp = df.copy()

    # Z-Score
    if metodo in ["zscore", "ambos"]:
        for col in colunas:
            z_col = f'z_score_{col}'
            df_temp[z_col] = zscore(df_temp[col].dropna())
        
        outliers_z = df_temp[
            [f'z_score_{col}' for col in colunas]
        ].apply(lambda x: (x.abs() > z_thres), axis=1).any(axis=1)
    else:
        outliers_z = pd.Series(False, index=df_temp.index)

    # IQR
    if metodo in ["iqr", "ambos"]:
        outliers_iqr = pd.Series(False, index=df_temp.index)
        for col in colunas:
            q1 = df_temp[col].quantile(0.25)
            q3 = df_temp[col].quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - iqr_factor * iqr
            limite_superior = q3 + iqr_factor * iqr
            outliers_iqr |= ~df_temp[col].between(limite_inferior, limite_superior) # (~) -> temos valores fora do intervalo
    else:
        outliers_iqr = pd.Series(False, index=df_temp.index)

    # Combina os índices dos outliers
    indices_outliers = df_temp[outliers_z | outliers_iqr].index

    # Remove os outliers
    df_filtrado = df_temp.drop(index=indices_outliers).copy()

    # Remove colunas auxiliares z_score_* se necessário
    if remover_zscore_colunas and metodo in ["zscore", "ambos"]:
        cols_z = [f'z_score_{col}' for col in colunas if f'z_score_{col}' in df_filtrado.columns]
        df_filtrado.drop(columns=cols_z, inplace=True)

    print(f"🔍 Outliers removidos: {len(indices_outliers)} linhas")
    return df_filtrado