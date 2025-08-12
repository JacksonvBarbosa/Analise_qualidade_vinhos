"""
Pipeline de classificação usando lazy loading.
"""
# Libs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.models.model_factory import ModelFactory
from src.models.classification.train_classification_model import train_model
from src.models.classification.predict_classification_model import predict
from src.models.classification.evaluate_classification_model import evaluate_classification
from src.models.save_load_model import save_model
from src.etl.extract import extract_csv_processed


def pipeline_classification(
    data_path,
    target_column,
    model_name='random_forest',  # Nome do modelo, não a instância
    custom_params=None,          # Parâmetros customizados
    scale_type=None,
    test_size=0.2,
    return_data=False,
    avarage='binary'
):
    """
    Pipeline de classificação com lazy loading.
    
    Args:
        data_path (str): Caminho dos dados, poder ser caminho do arquivo ou dataframe
        target_column (str): Nome da coluna target
        model_name (str): Nome do modelo ('random_forest', 'xgboost', etc.)
        custom_params (dict): Parâmetros customizados do modelo
        scale_type (str): Tipo de escalonamento ('standard', 'minmax')
        test_size (float): Proporção do teste
        
    Returns:
        dict: Resultados do pipeline
    
    Ler retorno:
        Inserir o nome da Variável e a coluna ex res['nome col']
    """
    
    print(f"Iniciando pipeline de classificação com modelo: {model_name}")
    
    # 1. Carregar dados
    if isinstance(data_path, str):
        df = extract_csv_processed(data_path)
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
    model = ModelFactory.create_classification_model(
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
    metrics = evaluate_classification(y_test, y_pred, average=avarage)
    
    print("\nMétricas:")
    for metric, value in metrics.items():
        if isinstance(value, list):
            print(f"  {metric}:")
            for row in value:
                print(f"    {row}")
        else:
            print(f"  {metric}: {value:.4f}")
    
    # 6. Salvar
    save_model(model, path="../models_storage", name=f"{model_name}_model.pkl")
    if scaler:
        save_model(scaler, path="../models_storage", name=f"{model_name}_scaler.pkl")
    
    return {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'model_name': model_name,
        'train_model': [model, X_train, X_test, y_train, y_test],
        'predict': y_pred
    }