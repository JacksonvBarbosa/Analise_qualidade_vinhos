"""
Pipeline de classificação para produção.
"""
# Libs
import pandas as pd
import joblib
from pathlib import Path

# Módulos do projeto
from analise_qualidade_vinhos.etl.extract import extract_csv_processed
from analise_qualidade_vinhos.config import DATA_PRODUCTION


def pipeline_production_classification(
    df_path_or_df,
    model_trained_path,
    scaler_trained_path=None,  # Agora opcional
    model_name='random_forest',
    save=False,
):
    """
    Pipeline de classificação para produção.
    
    Args:
        df_path_or_df (str ou pd.DataFrame): Caminho do arquivo ou DataFrame de entrada.
        model_trained_path (str): Caminho do modelo treinado (.pkl).
        scaler_trained_path (str, opcional): Caminho do scaler treinado (.pkl). 
                                                Se None, não aplica escalonamento.
        model_name (str): Nome do modelo.
        save (bool): Se True, salva o DataFrame com as previsões.
        
    Returns:
        pd.DataFrame: DataFrame com as colunas de previsões e probabilidades adicionadas (se disponíveis).
    """

    print(f"Iniciando pipeline de classificação com modelo: {model_name}")
    
    # 1. Carregar dados
    if isinstance(df_path_or_df, str):
        df = extract_csv_processed(df_path_or_df)
    else:
        df = df_path_or_df.copy()

    # Guardar os dados originais para salvar depois
    df_result = df.copy()

    # 2. Carregar scaler (opcional) e pré-processar
    if scaler_trained_path:
        try:
            scaler = joblib.load(scaler_trained_path)
            X_scaled = scaler.transform(df)
            print("✅ Scaler carregado e transformação aplicada com sucesso.")
        except Exception as e:
            print(f"❌ Erro ao carregar ou aplicar o scaler: {e}")
            X_scaled = df
    else:
        print("ℹ️ Nenhum scaler informado. Usando dados originais.")
        X_scaled = df

    # 3. Carregar o modelo
    try:
        model = joblib.load(model_trained_path)
        print("✅ Modelo treinado carregado com sucesso.")
    except Exception as e:
        print(f"❌ Erro ao carregar o modelo treinado: {e}")
        return None

    print(f"Iniciando predição...")
    
    # 4. Predições
    y_pred = model.predict(X_scaled)
    df_result['previsoes'] = y_pred

    # 5. Probabilidades (apenas se o modelo suportar)
    if hasattr(model, "predict_proba"):
        try:
            y_pred_proba = model.predict_proba(X_scaled)
            df_result['probabilidades'] = y_pred_proba[:, 1]  # Classe positiva
            print("✅ Probabilidades calculadas com sucesso.")
        except Exception as e:
            print(f"⚠️ Erro ao calcular probabilidades: {e}")
    else:
        print("ℹ️ O modelo não possui método predict_proba. Apenas previsões retornadas.")

    # 6. Salvar
    if save:
        output_path = Path(DATA_PRODUCTION) / f'{model_name}_production.xlsx'
        df_result.to_excel(output_path, index=False)
        print(f'✅ Arquivo salvo em: {output_path}')

    print('Pipeline concluído!')
    return df_result
