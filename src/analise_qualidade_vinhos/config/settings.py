import os

# Caminho da pasta onde este arquivo está
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho raiz do projeto (sobe 3 níveis)
BASE_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, "..", "..", ".."))

# Estrutura de pastas para dados
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_RAW = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED = os.path.join(DATA_DIR, "processed")
DATA_INTERIM = os.path.join(DATA_DIR, "interim")

# Pasta de notebooks
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebooks")

# Configurações padrão
SEED = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "target"

# Pastas de logs e modelos (dentro do projeto)
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Garante que existem
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


'''
🛠️ Modifique quando iniciar um projeto:

Mude TARGET_COLUMN para a variável dependente do seu dataset.

Adapte TEST_SIZE e SEED conforme necessidade.
'''