from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Calcula métricas acurácia, precisão, recall, F1_score, curva roc e matrix de confusão
def evaluate_classification(y_true, y_pred, y_proba=None, average='binary'):
    """
    Avalia métricas de classificação.

    Parâmetros:
        y_true (array): rótulos verdadeiros
        y_pred (array): rótulos previstos (classe)
        y_proba (array, opcional): probabilidades previstas (necessário para AUC)
        average (str): tipo de média para problemas multiclasses ('binary', 'macro', 'micro', 'weighted')

    Retorna:
        dicionário com métricas de avaliação
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    if y_proba is not None and average == 'binary':
        metrics["ROC AUC"] = roc_auc_score(y_true, y_proba)

    return metrics
