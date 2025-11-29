import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def detect_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5):
    """
    Detecta outliers em colunas numéricas usando o método IQR (Interquartile Range).
    Retorna um DataFrame com os índices dos outliers.
    """
    outliers = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
        outliers[col] = df[(df[col] < lower) | (df[col] > upper)].index.tolist()
    return outliers


def remove_outliers(df: pd.DataFrame, columns: list, factor: float = 1.5):
    """
    Remove outliers das colunas especificadas.
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df.reset_index(drop=True)


def summarize_dataframe(df: pd.DataFrame):
    """
    Retorna um resumo básico do DataFrame:
    - tipos
    - valores nulos
    - valores únicos
    - amostra de dados
    """
    summary = pd.DataFrame({
        "Tipo": df.dtypes,
        "Nulos (%)": df.isnull().mean() * 100,
        "Valores únicos": df.nunique()
    })
    print("\nResumo do DataFrame:\n")
    print(summary)
    print("\nAmostra dos dados:\n")
    print(df.head())

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

'''
🛠️ Modifique quando for iniciar um projeto:

Ajuste a função remove_outliers se quiser outro método (Z-score, IsolationForest etc.).

Pode adicionar funções específicas, como normalização de colunas personalizadas.
'''