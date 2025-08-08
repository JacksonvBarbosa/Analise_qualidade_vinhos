from sklearn.model_selection import train_test_split

def train_model(X, y, model, test_size=0.2, random_state=42, return_data=False):
    """
    Treina qualquer modelo de machine learning com base nos dados fornecidos.

    Parâmetros:
        X (pd.DataFrame ou np.ndarray): Features
        y (pd.Series ou np.ndarray): Target
        model: Instância do modelo (ex: LinearRegression(), RandomForestClassifier(), etc)
        test_size (float): Proporção dos dados para teste
        random_state (int): Semente para reprodutibilidade
        return_data (bool): Se True, retorna também os dados de treino/teste

    Retorna:
        model treinado
        (opcional) X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    
    model.fit(X_train, y_train)

    # retorna todas as variáveis mais o modelo treinado
    if return_data:
        return model, X_train, X_test, y_train, y_test
    
    return model
