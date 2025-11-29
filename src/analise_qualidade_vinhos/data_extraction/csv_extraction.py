import os
import pandas as pd
from analise_qualidade_vinhos.config.settings import DATA_RAW, DATA_PROCESSED, DATA_INTERIM

# Extrai dados Brutos
def extract_csv_raw(filename_or_path: str, sep: str = ',', thousands='.', decimal= ',', parse_dates=None, date_format=None, index_col=None, **kwargs) -> pd.DataFrame:
    """
    Lê CSV da pasta DATA_RAW ou de um caminho absoluto.
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_RAW, filename_or_path)
    return pd.read_csv(filepath, sep=sep, thousands=thousands, decimal=decimal, parse_dates=parse_dates, date_format=date_format, index_col=index_col, **kwargs)

# Extrai dados Processados
def extract_csv_processed(filename_or_path: str, sep: str = ',', parse_dates=None, date_format=None, index_col=None, **kwargs) -> pd.DataFrame:
    """
    Lê CSV da pasta DATA_PROCESSED ou de um caminho absoluto.
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_PROCESSED, filename_or_path)
    return pd.read_csv(filepath, sep=sep, **kwargs)

# Extrai dados Intermediários
def extract_csv_interim(filename_or_path: str, sep: str = ',', parse_dates=None, date_format=None, index_col=None, **kwargs) -> pd.DataFrame:
    """
    Lê CSV da pasta DATA_INTERIM ou de um caminho absoluto.
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_INTERIM, filename_or_path)
    return pd.read_csv(filepath, sep=sep, **kwargs)
