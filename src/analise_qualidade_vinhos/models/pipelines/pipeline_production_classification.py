"""
Pipeline de classificação usando lazy loading.
"""
# Libs
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from analise_qualidade_vinhos.models.classification.predict_classification_model import predict, predict_proba
from analise_qualidade_vinhos.etl.extract import extract_csv_processed

# Save
from analise_qualidade_vinhos.config import DATA_PRODUCTION
import joblib


def pipeline_production_classification(
    df,
    model_trained,
    model_name='random_forest',  # Nome do modelo, não a instância
    scale_type=None,
    save=None,
):
    """
    Pipeline de classificação para produção, antes de rodar
    os novos dados de entrada deverão ser tratados igual ao dados de treino
    
    Args:
        data_path (str): Caminho dos dados, poder ser caminho do arquivo ou dataframe
        target_column (str): Nome da coluna target
        model_name (str): Nome do modelo ('random_forest', 'xgboost', etc.)
        scale_type (str): Tipo de escalonamento ('standard', 'minmax')
        save (boolean): True salva modelo treinado para uso na produção
        
    Returns:
    
    Ler retorno:
        Inserir o nome da Variável e a coluna ex res['nome col']
    """

    print(f"Iniciando pipeline de classificação com modelo: {model_name}")
    
    # 1. Carregar dados
    if isinstance(df, str):
        df = extract_csv_processed(df)
    else:
        df = df

    X = df

    # 2. Pré-processamento
    scaler = None
    if scale_type == "standard":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scale_type == "minmax":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    model = joblib.load(model_trained)


    print(f"Iniciando predição do pipeline de classificação com modelo: {model_name}")
    # 3. Predições e avaliação
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # 4. Inseri Predições no DataFrame
    df['previsoes'] = y_pred
    df['probabilidades'] = y_pred_proba[:, 1]

    print(f"Salvando arquivo treinado de classificação do modelo: {model_name}")
    # 6. Salvar
    if save:
        if scaler:
            df.to_excel(f'{DATA_PRODUCTION}/{model_name}_production_scaler.xlsx')
        else:
            df.to_excel(f'{DATA_PRODUCTION}/{model_name}_production.xlsx')
    print('Arquivo salvo com sucesso!')