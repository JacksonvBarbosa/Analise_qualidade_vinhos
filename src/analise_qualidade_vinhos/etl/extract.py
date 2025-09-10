import os
import joblib
import pandas as pd
from analise_qualidade_vinhos.config import DATA_RAW, DATA_PROCESSED

def extract_csv_raw(filename: str, sep: str = ',') -> pd.DataFrame:
    filepath = f"{DATA_RAW}/{filename}"
    df = pd.read_csv(filepath, sep=sep)
    return df

def extract_csv_processed(filename: str, sep: str = ',') -> pd.DataFrame:
    filepath = f"{DATA_PROCESSED}/{filename}"
    df = pd.read_csv(filepath, sep=sep)
    return df

# Caminho base para os modelos (2 pastas acima do arquivo atual)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_STORAGE = os.path.join(BASE_DIR, "models_storage")

def extract_model(filename: str):
    """
    Carrega um modelo salvo no diretório models_storage.

    Args:
        filename (str): Nome do arquivo do modelo (ex.: 'lightgbm_model.pkl')

    Returns:
        object: modelo treinado carregado
    """
    filepath = os.path.join(MODELS_STORAGE, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modelo não encontrado em: {filepath}")

    model = joblib.load(filepath)
    return model
