import os
import json
from pyspark.sql import SparkSession
from analise_qualidade_vinhos.config.settings import DATA_RAW, DATA_PROCESSED, DATA_INTERIM

# Caminho correto para o config.json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # pasta atual (data_extraction)
CONFIG_PATH = os.path.join(BASE_DIR, "..", "utils", "config.json")  # volta 1 nível e entra em utils

# Inicializa sessão Spark (se não existir ainda)
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

def cria_sessao_spark(appname: str = None, modo: str = None, threads = None):
    '''
        função para criar sessão do Apache Spark.

        parametros:
        appname:  Nome dado a sessão criada,
        modo: Tipo de sessão 'local': para conexão local
    '''

    # Quantidade de cores a utilizar
    if threads is None:
        threads = 1
    else:
        threads = threads

    
    # Modo Local ou Master
    if modo is None:
        modo = 'yarn'
    elif modo == 'local':
        modo = f'local[{threads}]'
    else:
        modo = modo

    spark = SparkSession.builder \
        .appName(appname) \
        .master(modo) \
        .getOrCreate()
    return spark

# Função auxiliar para resolver o caminho do arquivo
def _resolve_path(base_dir: str, filename_or_path: str) -> str:
    return filename_or_path if os.path.isabs(filename_or_path) else os.path.join(base_dir, filename_or_path)

# Extrai dados Brutos
def extract_spark_csv_raw(spark, filename_or_path: str, sep: str = ',', header: bool = True, inferSchema: bool = True):
    """
    Lê CSV da pasta DATA_RAW ou de um caminho absoluto com Spark.
    """
    filepath = _resolve_path(DATA_RAW, filename_or_path)
    return spark.read.csv(filepath, sep=sep, header=header, inferSchema=inferSchema)

# Extrai dados Processados
def extract_spark_csv_processed(spark, filename_or_path: str, sep: str = ',', header: bool = True, inferSchema: bool = True):
    """
    Lê CSV da pasta DATA_PROCESSED ou de um caminho absoluto com Spark.
    """
    filepath = _resolve_path(DATA_PROCESSED, filename_or_path)
    return spark.read.csv(filepath, sep=sep, header=header, inferSchema=inferSchema)

# Extrai dados Intermediários
def extract_spark_csv_interim(spark, filename_or_path: str, sep: str = ',', header: bool = True, inferSchema: bool = True):
    """
    Lê CSV da pasta DATA_INTERIM ou de um caminho absoluto com Spark.
    """
    filepath = _resolve_path(DATA_INTERIM, filename_or_path)
    return spark.read.csv(filepath, sep=sep, header=header, inferSchema=inferSchema)
