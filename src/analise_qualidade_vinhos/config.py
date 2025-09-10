import os

# Caminho da pasta onde este arquivo está (ibovespa/)
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raiz do projeto (um nível acima de src/)
BASE_DIR = os.path.dirname(os.path.dirname(PACKAGE_DIR))

# Estrutura de pastas para dados
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_RAW = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED = os.path.join(DATA_DIR, "processed")
DATA_INTERIM = os.path.join(DATA_DIR, "interim")
DATA_PRODUCTION = os.path.join(DATA_DIR, "production")

# Pasta principal de notebooks
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebooks")
