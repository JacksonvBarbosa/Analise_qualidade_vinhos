# Arquivo: models/clustering/evaluate.py
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Avalia a qualidade dos clusters
def evaluate_clustering(X, labels):
    """
    Avalia clusters com métricas internas.

    Parâmetros:
        X (array): Dados de entrada
        labels (array): Labels atribuídos pelo modelo (clusters)

    Retorna:
        dicionário com métricas de clustering
    """
    return {
        "Silhouette Score": silhouette_score(X, labels),
        "Davies-Bouldin Index": davies_bouldin_score(X, labels)
    }