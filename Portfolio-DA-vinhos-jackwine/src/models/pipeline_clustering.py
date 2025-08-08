"""
Pipeline de clustering usando lazy loading.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.models.model_factory import ModelFactory
from src.models.clustering.train_clustering import train_clustering
from src.models.clustering.predict_clustering import predict
from src.models.clustering.evaluate_clustering import evaluate_clustering
from src.models.save_load_model import save_model


def pipeline_clustering(
    data_path,
    model_name='kmeans',      # Nome do modelo, não a instância
    custom_params=None,       # Parâmetros customizados
    scale_type=None,          # 'standard', 'minmax' ou None
    n_clusters=3              # Número de clusters (para modelos que precisam)
):
    """
    Pipeline de clustering com lazy loading.
    
    Args:
        data_path (str): Caminho dos dados
        model_name (str): Nome do modelo ('kmeans', 'dbscan', 'gaussian_mixture', etc.)
        custom_params (dict): Parâmetros customizados do modelo
        scale_type (str): Tipo de escalonamento ('standard', 'minmax', None)
        n_clusters (int): Número de clusters (usado se não estiver em custom_params)
        
    Returns:
        dict: Resultados do pipeline

    Ler retorno:
        Inserir o nome da Variável e a coluna ex res['nome col']
    """
    
    print(f"Iniciando pipeline de clustering com modelo: {model_name}")
    
    # 1. Carregar dados
    if isinstance(data_path, str):
        df = pd.read_csv(data_path)
    else:
        df = data_path
    X = df.copy()  # Clustering não tem coluna target
    
    # 2. Pré-processamento
    scaler = None
    if scale_type == "standard":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scale_type == "minmax":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    
    # 3. Preparar parâmetros do modelo
    # Se n_clusters for passado e modelo suportar, adicionar aos custom_params
    if custom_params is None:
        custom_params = {}
    
    # Adicionar n_clusters se o modelo precisar e não foi especificado
    if model_name in ['kmeans', 'gaussian_mixture'] and 'n_clusters' not in custom_params and 'n_components' not in custom_params:
        if model_name == 'kmeans':
            custom_params['n_clusters'] = n_clusters
        elif model_name == 'gaussian_mixture':
            custom_params['n_components'] = n_clusters
    
    # 4. Criar modelo usando Factory (LAZY LOADING)
    model = ModelFactory.create_clustering_model(
        model_name=model_name, 
        custom_params=custom_params
    )
    print(f"Modelo {model_name} criado com sucesso!")
    
    # 5. Treinar
    model, X_train = train_clustering(X, model=model)
    
    # 6. Predições e avaliação
    y_pred = predict(model, X_train)
    metrics = evaluate_clustering(X_train, y_pred)
    
    print("Métricas:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 7. Salvar
    save_model(model, path="../models_storage", name=f"{model_name}_model.pkl")
    if scaler:
        save_model(scaler, path="../models_storage", name=f"{model_name}_scaler.pkl")
    
    return {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'model_name': model_name,
        'predictions': y_pred
    }


