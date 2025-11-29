import os
import pandas as pd
from analise_qualidade_vinhos.config.settings import DATA_RAW, DATA_PROCESSED, DATA_INTERIM

def extract_excel_raw(filename_or_path: str, sheet_name: str = None, **kwargs) -> pd.DataFrame:
    """
    Lê Excel (.xlsx, .xls) da pasta DATA_RAW ou de um caminho absoluto.
    
    Args:
        filename_or_path: Nome do arquivo ou caminho absoluto
        sheet_name: Nome da planilha (None = primeira planilha)
        **kwargs: Parâmetros adicionais para pd.read_excel()
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_RAW, filename_or_path)
    return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)

def extract_excel_processed(filename_or_path: str, sheet_name: str = None, **kwargs) -> pd.DataFrame:
    """
    Lê Excel (.xlsx, .xls) da pasta DATA_PROCESSED ou de um caminho absoluto.
    
    Args:
        filename_or_path: Nome do arquivo ou caminho absoluto
        sheet_name: Nome da planilha (None = primeira planilha)
        **kwargs: Parâmetros adicionais para pd.read_excel()
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_PROCESSED, filename_or_path)
    return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)

def extract_excel_interim(filename_or_path: str, sheet_name: str = None, **kwargs) -> pd.DataFrame:
    """
    Lê Excel (.xlsx, .xls) da pasta DATA_INTERIM ou de um caminho absoluto.
    
    Args:
        filename_or_path: Nome do arquivo ou caminho absoluto
        sheet_name: Nome da planilha (None = primeira planilha)
        **kwargs: Parâmetros adicionais para pd.read_excel()
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_INTERIM, filename_or_path)
    return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)