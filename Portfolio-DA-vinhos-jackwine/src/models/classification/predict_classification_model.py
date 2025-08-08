# Previsionamento do modelo
def predict(model, X):
    """
    Realiza predições com o modelo treinado.

    Parâmetros:
        model: Modelo já treinado
        X: Dados de entrada

    Retorna:
        array com predições
    """
    return model.predict(X)