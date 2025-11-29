import os
import pandas as pd
from analise_qualidade_vinhos.config.settings import DATA_RAW, DATA_PROCESSED, DATA_INTERIM

def extract_parquet_raw(filename_or_path: str, **kwargs) -> pd.DataFrame:
    """
    Lê Parquet da pasta DATA_RAW ou de um caminho absoluto.
    
    Args:
        filename_or_path: Nome do arquivo ou caminho absoluto
        **kwargs: Parâmetros adicionais para pd.read_parquet()
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_RAW, filename_or_path)
    return pd.read_parquet(filepath, **kwargs)

def extract_parquet_processed(filename_or_path: str, **kwargs) -> pd.DataFrame:
    """
    Lê Parquet da pasta DATA_PROCESSED ou de um caminho absoluto.
    
    Args:
        filename_or_path: Nome do arquivo ou caminho absoluto
        **kwargs: Parâmetros adicionais para pd.read_parquet()
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_PROCESSED, filename_or_path)
    return pd.read_parquet(filepath, **kwargs)

def extract_parquet_interim(filename_or_path: str, **kwargs) -> pd.DataFrame:
    """
    Lê Parquet da pasta DATA_INTERIM ou de um caminho absoluto.
    
    Args:
        filename_or_path: Nome do arquivo ou caminho absoluto
        **kwargs: Parâmetros adicionais para pd.read_parquet()
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_INTERIM, filename_or_path)
    return pd.read_parquet(filepath, **kwargs)