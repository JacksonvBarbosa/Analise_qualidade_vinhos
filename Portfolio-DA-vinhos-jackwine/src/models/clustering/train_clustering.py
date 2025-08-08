# Arquivo: models/clustering/train.py
from sklearn.base import ClusterMixin

# Treinamento para modelos de clustering (sem y)
def train_clustering(X, model: ClusterMixin):
    """
    Treina modelo de clustering.

    Parâmetros:
        X (array-like): Dados de entrada
        model (ClusterMixin): Modelo de clustering (ex: KMeans, DBSCAN)

    Retorna:
        model treinado
        y_pred: clusters previstos
    """
    y_pred = model.fit_predict(X)
    return model, y_pred