import os
import pandas as pd
from analise_qualidade_vinhos.config.settings import DATA_PROCESSED, DATA_INTERIM, DATA_RAW 

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
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    
    # Monta o caminho completo
    filepath = os.path.join(DATA_PROCESSED, filename)
    
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
    os.makedirs(DATA_INTERIM, exist_ok=True)
    
    # Monta o caminho completo
    filepath = os.path.join(DATA_INTERIM, filename)
    
    # Salva usando o caminho completo (não apenas o filename!)
    df.to_csv(filepath, index=index)  # ✅ Usa filepath, não filename
    
    print(f"✅ Arquivo salvo em: {filepath}")
    return filepath

# Função adicional para carregar CSVs
def load_from_csv_processed(filename: str, parse_dates: str = None):
    """
    Carrega um CSV da pasta DATA_PROCESSED.
    
    Args:
        filename: Nome do arquivo
        parse_dates = ['col'] insira o nome da coluna
    
    Returns:
        DataFrame do pandas
    """
    
    filepath = os.path.join(DATA_PROCESSED, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {filepath}")
    
    if parse_dates == None:
        return pd.read_csv(filepath)
    else:
        return pd.read_csv(filepath, parse_dates=parse_dates)
    

# Função adicional para carregar CSVs
def load_from_csv_raw(filename: str, parse_dates: str = None):
    """
    Carrega um CSV da pasta DATA_PROCESSED.
    
    Args:
        filename: Nome do arquivo
        parse_dates = ['col'] insira o nome da coluna
    
    Returns:
        DataFrame do pandas
    """
    
    filepath = os.path.join(DATA_RAW, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {filepath}")
    
    if parse_dates == None:
        return pd.read_csv(filepath)
    else:
        return pd.read_csv(filepath, parse_dates=parse_dates)