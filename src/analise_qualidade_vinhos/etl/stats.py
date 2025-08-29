# src/etl/stats.py

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, shapiro, normaltest, zscore

# Calcular skewness e Kurtosis
def calcula_skew_kurtosis(df: pd.DataFrame):
    """Retorna o skewness e kurtosis de cada coluna numérica do dataframe."""
    skewness = df.skew(numeric_only=True)
    kurtosis = df.kurtosis(numeric_only=True)
    return skewness, kurtosis

# Obter dados nulos
def obter_dados_nulos(df: pd.DataFrame):
    """Retorna a quantidade e a porcentagem de valores nulos em cada coluna."""
    total_missing = df.isnull().sum()
    percent_missing = (total_missing / len(df)) * 100
    return pd.DataFrame({
        'Valores Nulos': total_missing,
        '% Nulos': percent_missing
    }).sort_values(by='Valores Nulos', ascending=False)

# Detectar Outliers
def detecta_outliers_iqr(df: pd.DataFrame, colunas: str):
    # Z-Score e Intervalo Interquartil (IQR)

    res_irq = 0
    res_z = 0

    for col in colunas:     
        print(f'\n📌 Analisando a coluna: {col.upper()}')

        # === Z-SCORE ===
        df[f'z_score_{col}'] = zscore(df[col].dropna())  # evita erro com NaN
        outliers_z = df[(df[f'z_score_{col}'] > 3) | (df[f'z_score_{col}'] < -3)] # 3 vezes acima ou abaixo do desvio padrão
        print(f'🔹 Outliers (Z-Score): {len(outliers_z)}')
        print(outliers_z[[col, f'z_score_{col}']].head())

        # === IQR ===
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR # valores abaixo de 1.5 vezes do Intervalo interquartil
        limite_superior = Q3 + 1.5 * IQR # valores acima de 1.5 vezes do Intervalo interquartil

        outliers_iqr = df[(df[col] < limite_inferior) | (df[col] > limite_superior)] # Efetua a condição para definir se é outlier ou não
        print(f'🔸 Outliers (IQR): {len(outliers_iqr)}')
        print(outliers_iqr[[col]].head())
        res_irq += 1
        res_z += 1
    return res_irq, res_z



# Matrix de correlação
def matrix_correlacao(df: pd.DataFrame):
    """Retorna a matriz de correlação entre colunas numéricas."""
    return df.corr(numeric_only=True)


# Teste estatístico de skewness, Kurtosis, Shapiro-Wilk e D'Agostino
def analisar_distribuicao(df: pd.DataFrame, coluna: str) -> None:
    serie = df[coluna].dropna()

    # Subamostra para o Shapiro-Wilk se houver mais de 500 valores
    amostra_shapiro = serie.sample(n=500, random_state=42) if len(serie) > 500 else serie

    # Estatísticas
    assimetria = skew(serie)
    curtose = kurtosis(serie, fisher=True)
    stat_shapiro, p_shapiro = shapiro(amostra_shapiro)
    stat_dagostino, p_dagostino = normaltest(serie)

    # Exibição formatada
    print(f"📊 Análise da coluna: **{coluna}**\n")
    print(f"📈 Assimetria: {assimetria:.4f} {'(positiva)' if assimetria > 0 else '(negativa)' if assimetria < 0 else '(simétrica)'}")
    print(f"📉 Curtose: {curtose:.4f} {'(leptocúrtica)' if curtose > 0 else '(platicúrtica)' if curtose < 0 else '(mesocúrtica)'}\n")
    
    print(f"🧪 Teste de Shapiro-Wilk:")
    print(f"   Estatística: {stat_shapiro:.4f}")
    print(f"   p-valor: {p_shapiro:.4f} {'→ ❌ Não normal' if p_shapiro < 0.05 else '→ ✅ Normal'}\n")
    
    print(f"🧪 Teste de D’Agostino-Pearson:")
    print(f"   Estatística: {stat_dagostino:.4f}")
    print(f"   p-valor: {p_dagostino:.4f} {'→ ❌ Não normal' if p_dagostino < 0.05 else '→ ✅ Normal'}")
