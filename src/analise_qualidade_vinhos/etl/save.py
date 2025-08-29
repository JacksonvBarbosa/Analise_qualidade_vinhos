import os
from analise_qualidade_vinhos.config import DATA_PROCESSED

def save_to_csv(df, filename: str):
    filepath = os.path.join(DATA_PROCESSED, filename)
    df.to_csv(filename, index=False)
