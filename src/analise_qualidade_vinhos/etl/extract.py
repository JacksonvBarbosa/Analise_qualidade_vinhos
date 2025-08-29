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