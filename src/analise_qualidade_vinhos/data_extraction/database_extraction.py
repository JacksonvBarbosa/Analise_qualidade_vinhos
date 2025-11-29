import pandas as pd
import sqlite3
from typing import Optional, Dict, Any
from sqlalchemy import create_engine

def extract_from_sqlite(
    database_path: str,
    query: str,
    params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Extrai dados de banco SQLite usando uma query SQL.
    
    Args:
        database_path: Caminho para o arquivo .db
        query: Query SQL para execução
        params: Parâmetros para a query (opcional)
    """
    try:
        with sqlite3.connect(database_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        raise Exception(f"Erro ao conectar com SQLite: {e}")

def extract_from_database(
    connection_string: str,
    query: str,
    params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Extrai dados de qualquer banco suportado pelo SQLAlchemy.
    
    Args:
        connection_string: String de conexão SQLAlchemy
            Ex: 'postgresql://user:pass@localhost/db'
            Ex: 'mysql://user:pass@localhost/db'  
            Ex: 'sqlite:///path/to/db.sqlite'
        query: Query SQL para execução
        params: Parâmetros para a query (opcional)
    """
    try:
        engine = create_engine(connection_string)
        return pd.read_sql_query(query, engine, params=params)
    except Exception as e:
        raise Exception(f"Erro ao conectar com banco de dados: {e}")

def extract_table_from_database(
    connection_string: str,
    table_name: str,
    schema: Optional[str] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Extrai uma tabela completa do banco de dados.
    
    Args:
        connection_string: String de conexão SQLAlchemy
        table_name: Nome da tabela
        schema: Schema da tabela (opcional)
        limit: Limite de registros (opcional)
    """
    query = f"SELECT * FROM {table_name}"
    if schema:
        query = f"SELECT * FROM {schema}.{table_name}"
    if limit:
        query += f" LIMIT {limit}"
    
    return extract_from_database(connection_string, query)