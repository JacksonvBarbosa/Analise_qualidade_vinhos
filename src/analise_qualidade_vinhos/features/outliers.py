import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Substitui os valores do outlier pelos extremos superiores e inferiores dos dados
def cap_outliers(df, columns=None, factor=1.5, method='iqr', verbose=True):
    """
    Aplica capping nos outliers usando IQR ou Z-score
    
    Args:
        df: DataFrame
        columns: colunas para aplicar (None = todas numéricas)
        factor: fator multiplicador (1.5 para IQR, 3 para Z-score)
        method: 'iqr' ou 'zscore'
        verbose: mostrar informações do tratamento
    
    Returns:
        DataFrame com outliers tratados
    """
    df_capped = df.copy()
    
    if columns is None:
        columns = df_capped.select_dtypes(include=np.number).columns
    
    outliers_info = {}
    
    for col in columns:
        if method == 'iqr':
            # Método IQR (seu método original)
            Q1 = df_capped[col].quantile(0.25)
            Q3 = df_capped[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - factor * IQR
            upper_limit = Q3 + factor * IQR
            
        elif method == 'zscore':
            # Método Z-score
            mean = df_capped[col].mean()
            std = df_capped[col].std()
            lower_limit = mean - factor * std
            upper_limit = mean + factor * std
        
        else:
            raise ValueError("method deve ser 'iqr' ou 'zscore'")
        
        # Contar outliers antes do tratamento
        outliers_lower = (df_capped[col] < lower_limit).sum()
        outliers_upper = (df_capped[col] > upper_limit).sum()
        total_outliers = outliers_lower + outliers_upper
        
        # Aplicar capping
        df_capped[col] = np.where(df_capped[col] < lower_limit, lower_limit, df_capped[col])
        df_capped[col] = np.where(df_capped[col] > upper_limit, upper_limit, df_capped[col])
        
        # Salvar informações
        outliers_info[col] = {
            'outliers_lower': outliers_lower,
            'outliers_upper': outliers_upper,
            'total_outliers': total_outliers,
            'percentage': (total_outliers / len(df_capped)) * 100,
            'lower_limit': lower_limit,
            'upper_limit': upper_limit
        }
    
    if verbose:
        print("=== RELATÓRIO DE TRATAMENTO DE OUTLIERS ===")
        for col, info in outliers_info.items():
            if info['total_outliers'] > 0:
                print(f"🔧 {col}:")
                print(f"   Outliers tratados: {info['total_outliers']} ({info['percentage']:.2f}%)")
                print(f"   Limites: [{info['lower_limit']:.2f}, {info['upper_limit']:.2f}]")
        print("="*50)
    
    return df_capped


# Detectar outlier mais não exclui
def detect_outliers(df, columns=None, method='iqr', factor=1.5):
    """
    Detecta outliers sem fazer tratamento
    
    Args:
        df: DataFrame
        columns: colunas para analisar
        method: 'iqr', 'zscore' ou 'isolation'
        factor: fator multiplicador
    
    Returns:
        DataFrame com informações dos outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns
    
    outliers_report = []
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - factor * IQR
            upper_limit = Q3 + factor * IQR
            outliers_mask = (df[col] < lower_limit) | (df[col] > upper_limit)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers_mask = z_scores > factor
            lower_limit = df[col].mean() - factor * df[col].std()
            upper_limit = df[col].mean() + factor * df[col].std()
        
        outliers_count = outliers_mask.sum()
        outliers_percentage = (outliers_count / len(df)) * 100
        
        outliers_report.append({
            'coluna': col,
            'outliers_count': outliers_count,
            'outliers_percentage': outliers_percentage,
            'lower_limit': lower_limit,
            'upper_limit': upper_limit,
            'min_value': df[col].min(),
            'max_value': df[col].max()
        })
    
    return pd.DataFrame(outliers_report)


# Remove outliers
def remove_outliers(df, columns=None, method='iqr', factor=1.5, verbose=True):
    """
    Remove outliers completamente (linhas inteiras)
    
    Args:
        df: DataFrame
        columns: colunas para considerar na remoção
        method: 'iqr' ou 'zscore'
        factor: fator multiplicador
        verbose: mostrar informações
    
    Returns:
        DataFrame sem outliers, índices removidos
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns
    
    df_clean = df.copy()
    outliers_indices = set()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - factor * IQR
            upper_limit = Q3 + factor * IQR
            col_outliers = df_clean[(df_clean[col] < lower_limit) | 
                                    (df_clean[col] > upper_limit)].index
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            col_outliers = df_clean[z_scores > factor].index
        
        outliers_indices.update(col_outliers)
    
    # Remover outliers
    df_clean = df_clean.drop(index=list(outliers_indices))
    
    if verbose:
        removed_count = len(outliers_indices)
        removed_percentage = (removed_count / len(df)) * 100
        print(f"🗑️  Linhas removidas: {removed_count} ({removed_percentage:.2f}%)")
        print(f"📊 Dataset original: {len(df)} → Dataset limpo: {len(df_clean)}")
    
    return df_clean, list(outliers_indices)


# Plota gráficos dos outliers e sem os outliers
def plot_outliers_comparison(df_original, df_treated, columns=None, figsize=(15, 10)):
    """
    Plota comparação antes/depois do tratamento de outliers
    
    Args:
        df_original: DataFrame original
        df_treated: DataFrame após tratamento
        columns: colunas para plotar
        figsize: tamanho da figura
    """
    if columns is None:
        columns = df_original.select_dtypes(include=np.number).columns
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(columns[:n_rows * n_cols]):
        row, col_idx = divmod(i, n_cols)
        ax = axes[row, col_idx]
        
        # Boxplot comparativo
        data_to_plot = [df_original[col].dropna(), df_treated[col].dropna()]
        bp = ax.boxplot(data_to_plot, labels=['Original', 'Tratado'], patch_artist=True)
        
        # Colorir boxplots
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(f'{col}', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Remover subplots vazios
    for i in range(len(columns), n_rows * n_cols):
        row, col_idx = divmod(i, n_cols)
        fig.delaxes(axes[row, col_idx])
    
    plt.tight_layout()
    plt.suptitle('Comparação: Antes vs Depois do Tratamento de Outliers', 
                fontsize=14, y=1.02)
    plt.show()