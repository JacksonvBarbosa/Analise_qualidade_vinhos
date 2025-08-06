import os

# Variáveis dos Diretórios
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
DATA_INTERIM = os.path.join(BASE_DIR, 'data', 'interim')