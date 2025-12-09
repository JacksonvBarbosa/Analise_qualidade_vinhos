"""Data loading and simple validation helpers."""

from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from analise_qualidade_vinhos.config.settings import (
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    RAW_DATA_PATH
)
from analise_qualidade_vinhos.features.engineering import build_feature_matrix


def load_raw_data(path: Union[Path, str, None] = None, sep: str = ';') -> pd.DataFrame:
    """Carrega um CSV bruto com os dados do vinho.

    Parâmetros:
    - path: caminho para o arquivo (`Path` ou `str`). Se `None`, usa `RAW_DATA_PATH`.
    - sep: separador do CSV (padrão '`;`' para o dataset UCI de vinho).

    Retorna:
    - `pd.DataFrame` com os dados carregados.
    """
    # Usa o caminho padrão se nenhum for informado
    if path is None:
        path = RAW_DATA_PATH

    # Converte str para Path
    if isinstance(path, str):
        path = Path(path)

    if not isinstance(path, Path):
        raise TypeError("path deve ser do tipo str, Path ou None")

    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em {path}")

    # UCI wine dataset usa ';' como separador
    return pd.read_csv(path, sep=sep)


def load_featured_data(path: Union[Path, str, None] = None) -> pd.DataFrame:
    """Carrega os dados brutos e aplica `build_feature_matrix`.

    Aceita caminho como `str`, `Path` ou `None` (usa arquivo padrão).
    """
    raw = load_raw_data(path)
    featured = build_feature_matrix(raw)
    return featured


def train_test_split_featured(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
)

