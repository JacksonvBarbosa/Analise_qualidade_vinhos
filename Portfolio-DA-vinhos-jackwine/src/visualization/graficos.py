def grafo_barra(df, x, y, hue, paleta='tab10', titulo='', ylabel='', xlabel=''):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10,6))

    ax = sns.barplot(data=df, x=x,y=y,hue=hue , palette=paleta)
    plt.title(titulo, fontsize=18, fontweight='bold', loc='left')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=360)

    y_limite = y.max()
    plt.ylim(0, y_limite * 1.1)

    for container in ax.containers:
        ax.bar_label(container, label_type='edge', padding=3)

    plt.show()

def grafo_heatmap(df, annot=True, tam_linha=0.5, fmt='.2f'):
    import seaborn as sns
    import matplotlib.pyplot as plt

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10,6))
    sns.heatmap(corr, annot=annot, linewidths=tam_linha, fmt=fmt)

    plt.show()