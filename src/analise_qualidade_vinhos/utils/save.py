import os
import pandas as pd
from analise_qualidade_vinhos.config.settings import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_PATH 

# Salva em Processed
def save_to_csv_processed(df, filename: str, index=None):
    """
    Salva um DataFrame como CSV na pasta DATA_PROCESSED.
    
    Args:
        df: DataFrame do pandas
        filename: Nome do arquivo (ex: 'dados.csv')
    
    Returns:
        str: Caminho completo do arquivo salvo
    """
    # Garante que a pasta existe
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Monta o caminho completo
    filepath = os.path.join(PROCESSED_DATA_DIR, filename)
    
    # Salva usando o caminho completo (não apenas o filename!)
    df.to_csv(filepath, index=index)  # ✅ Usa filepath, não filename
    
    print(f"✅ Arquivo salvo em: {filepath}")
    return filepath

# Salva em Interim
def save_to_csv_interim(df, filename: str, index=None):
    """
    Salva um DataFrame como CSV na pasta DATA_PROCESSED.
    
    Args:
        df: DataFrame do pandas
        filename: Nome do arquivo (ex: 'dados.csv')
    
    Returns:
        str: Caminho completo do arquivo salvo
    """
    # Garante que a pasta existe
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    
    # Monta o caminho completo
    filepath = os.path.join(INTERIM_DATA_DIR, filename)
    
    # Salva usando o caminho completo (não apenas o filename!)
    df.to_csv(filepath, index=index)  # ✅ Usa filepath, não filename
    
    print(f"✅ Arquivo salvo em: {filepath}")
    return filepath

# Função adicional para carregar CSVs
def load_from_csv_processed(filename: str, parse_dates: str = None):
    """
    Carrega um CSV da pasta PROCESSED_DATA_DIR.
    
    Args:
        filename: Nome do arquivo
        parse_dates = ['col'] insira o nome da coluna
    
    Returns:
        DataFrame do pandas
    """
    
    filepath = os.path.join(PROCESSED_DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {filepath}")
    
    if parse_dates == None:
        return pd.read_csv(filepath)
    else:
        return pd.read_csv(filepath, parse_dates=parse_dates)
    

# Função adicional para carregar CSVs
def load_from_csv_raw(filename: str, parse_dates: str = None):
    """
    Carrega um CSV da pasta RAW_DATA_PATH.
    
    Args:
        filename: Nome do arquivo
        parse_dates = ['col'] insira o nome da coluna
    
    Returns:
        DataFrame do pandas
    """
    
    filepath = os.path.join(RAW_DATA_PATH, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {filepath}")
    
    if parse_dates == None:
        return pd.read_csv(filepath)
    else:
        return pd.read_csv(filepath, parse_dates=parse_dates)