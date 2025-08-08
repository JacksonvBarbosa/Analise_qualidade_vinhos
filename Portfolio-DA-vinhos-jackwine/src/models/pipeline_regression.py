"""
Pipeline de regressão usando lazy loading.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.models.model_factory import ModelFactory
from src.models.regression.train_regression_model import train_model
from src.models.regression.predict_regression_model import predict
from src.models.regression.evaluate_regression_model import evaluate_regression
from src.etl.extract import extract_csv_processed
from src.models.save_load_model import save_model


def pipeline_regression(
    data_path,
    target_column,
    model_name='linear_regression',  # Nome do modelo, não a instância
    custom_params=None,              # Parâmetros customizados
    scale_type=None,                 # 'standard', 'minmax' ou None
    test_size=0.2,
    return_data=False
):
    """
    Pipeline de regressão com lazy loading.
    
    Args:
        data_path (str): Caminho dos dados
        target_column (str): Nome da coluna target
        model_name (str): Nome do modelo ('linear_regression', 'random_forest', 'xgboost', etc.)
        custom_params (dict): Parâmetros customizados do modelo
        scale_type (str): Tipo de escalonamento ('standard', 'minmax', None)
        test_size (float): Proporção do teste
        
    Returns:
        dict: Resultados do pipeline
    """
    
    print(f"Iniciando pipeline de regressão com modelo: {model_name}")
    
    # 1. Carregar dados
    if isinstance(data_path, str):
        df = pd.read_csv(data_path)
    else:
        df = data_path
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 2. Pré-processamento
    scaler = None
    if scale_type == "standard":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scale_type == "minmax":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    
    # 3. Criar modelo usando Factory (LAZY LOADING)
    model = ModelFactory.create_regression_model(
        model_name=model_name, 
        custom_params=custom_params
    )
    print(f"Modelo {model_name} criado com sucesso!")
    
    # 4. Treinar
    model, X_train, X_test, y_train, y_test = train_model(
        X, y, model=model, test_size=test_size, return_data=return_data
    )
    
    # 5. Predições e avaliação
    y_pred = predict(model, X_test)
    metrics = evaluate_regression(y_test, y_pred)
    
    print("Métricas:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 6. Salvar
    save_model(model, path="models_storage", name=f"{model_name}_model.pkl")
    if scaler:
        save_model(scaler, path="models_storage", name=f"{model_name}_scaler.pkl")
    
    return {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'model_name': model_name
    }