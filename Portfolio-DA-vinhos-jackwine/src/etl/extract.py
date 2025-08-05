import pandas as pd
from src.config import DATA_RAW

def extract_csv(filename: str, sep: str = ',') -> pd.DataFrame:
    filepath = f"{DATA_RAW}/{filename}"
    df = pd.read_csv(filepath, sep=sep)
    return df

