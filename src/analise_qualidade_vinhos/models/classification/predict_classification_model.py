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

def predict_proba(model, X):
    '''
    Realiza predições de probabilidade com o modelo treinado.

    Parâmetros:
        model: Modelo já treinado
        X: Dados de entrada

    Retorna:
        array com predições da probabilidade
    '''
    return model.predict_proba(X)