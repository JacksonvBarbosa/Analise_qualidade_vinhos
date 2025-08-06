# Libs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico de Barras
def grafo_barra(df: pd.DataFrame, x, y, hue, paleta='tab10', titulo='', ylabel='', xlabel=''):
    """
    Esta função recebe parametros para gerar gráfico de barra.
    
    Parâmetros:
    df: dataframe
    x: coluna do df no eixo x (horizontal)
    y: coluna do df no eixo y (vertical)
    hue: para agrupar os dados categoricas e atribuir cores diferentes a elas
    titulo: titulo do gráfico
    xlabel: rótulo do eixo x
    ylabel: rótulo do eixo y

    Retorna:
    Gráfico de barras (barplot)
    """
    plt.figure(figsize=(10,6))

    ax = sns.barplot(df, x=x,y=y,hue=hue , palette=paleta)
    plt.title(titulo, fontsize=18, fontweight='bold', loc='left')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=360)

    y_limite = y.max()
    plt.ylim(0, y_limite * 1.1)

    for container in ax.containers:
        ax.bar_label(container, label_type='edge', padding=3)

    plt.show()

# Gráfico de Mapa de Calor
def grafo_heatmap(df: pd.DataFrame, annot=True, tam_linha=0.5, fmt='.2f'):
    """
    Esta função recebe parametros para gerar gráfico de mapa de calor para correlações.
    
    Parâmetros:
    df: dataframe
    annot: para inserir os valores no gráfico
    tam_linha: espaço entre um bloco e outro
    fmt: arredonda casa decimal após a virgula

    Retorna:
    Gráfico de Mapa de Calor (heatmap)
    """
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10,6))
    sns.heatmap(corr, annot=annot, linewidths=tam_linha, fmt=fmt)

    plt.show()

# Gráfico de Dispersão (Scatterplot)
def grafo_scatterplot(df: pd.DataFrame, x, y, color='blue', alpha=0.7, titulo='', ylabel='', xlabel=''):
    """
    Esta função recebe parametros para gerar gráfico de barra.
    
    Parâmetros:
    df: dataframe
    x: coluna do df no eixo x (horizontal)
    y: coluna do df no eixo y (vertical)
    color: cor dos pontos no gráfico
    alpha: transpârencia dos pontos no gráfico 
    titulo: titulo do gráfico
    xlabel: rótulo do eixo x
    ylabel: rótulo do eixo y

    Retorna:
    Gráfico de de dispersão (scatterplot)
    """
    plt.figure(figsize=(10, 6))

    sns.scatterplot(data=df, x=x, y=y, color=color, alpha=alpha)
    sns.regplot(data=df, x=x, y=y, 
                scatter=False, line_kws={'color': 'red', 'linewidth': 2})  # Linha de regressão
    plt.title(titulo, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()


# Gráfico de Distribuição - histograma
def grafo_distribuicao(df: pd.DataFrame, column: str, kde=True, bins=30, color='purple'):
    """
    Plota a distribuição de uma coluna numérica.
    
    Parâmetros:
    df: dataframe
    column: coluna do dataframe
    kde: Kernel Density Estimation(Estimativa de Densidade por Kernel) - mostra uma curva suave que
    bins: intervalos que os dados serão divididos no histograma

    Retorna:
    Gráfico de distribuição (histplot)
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=kde, bins=bins, color=color)
    plt.title(f'Distribuição de {column}')
    plt.xlabel(column)
    plt.ylabel('Frequência')
    plt.show()

# Grráfico Boxplot
def grafo_boxplot(df: pd.DataFrame, column: str):
    """Plota um boxplot para uma coluna numérica."""
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot de {column}')
    plt.show()