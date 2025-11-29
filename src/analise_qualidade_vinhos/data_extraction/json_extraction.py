import os
import json
import pandas as pd
from analise_qualidade_vinhos.config.settings import DATA_RAW, DATA_PROCESSED, DATA_INTERIM

def extract_json_raw(filename_or_path: str) -> pd.DataFrame:
    """
    Lê JSON da pasta DATA_RAW ou de um caminho absoluto e retorna como DataFrame.
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_RAW, filename_or_path)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.json_normalize(data)

def extract_json_processed(filename_or_path: str) -> pd.DataFrame:
    """
    Lê JSON da pasta DATA_PROCESSED ou de um caminho absoluto e retorna como DataFrame.
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_PROCESSED, filename_or_path)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.json_normalize(data)

def extract_json_interim(filename_or_path: str) -> pd.DataFrame:
    """
    Lê JSON da pasta DATA_INTERIM ou de um caminho absoluto e retorna como DataFrame.
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_INTERIM, filename_or_path)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.json_normalize(data)
