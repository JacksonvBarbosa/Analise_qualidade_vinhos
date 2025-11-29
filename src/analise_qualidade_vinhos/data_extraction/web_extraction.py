import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, List
from urllib.parse import urljoin

def extract_table_from_html(
    url: str,
    table_index: int = 0,
    headers: Optional[Dict[str, str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Extrai tabelas HTML de uma página web.
    
    Args:
        url: URL da página
        table_index: Índice da tabela (0 = primeira tabela)
        headers: Headers HTTP para a requisição
        **kwargs: Parâmetros adicionais para pd.read_html()
    """
    try:
        tables = pd.read_html(url, **kwargs)
        if table_index >= len(tables):
            raise IndexError(f"Tabela {table_index} não encontrada. Encontradas {len(tables)} tabelas.")
        return tables[table_index]
    except Exception as e:
        raise Exception(f"Erro ao extrair tabela HTML: {e}")

def extract_from_web_scraping(
    url: str,
    css_selector: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30
) -> List[str]:
    """
    Extrai dados de uma página web usando web scraping com BeautifulSoup.
    
    Args:
        url: URL da página
        css_selector: Seletor CSS para extrair elementos
        headers: Headers HTTP para a requisição
        timeout: Timeout em segundos
    
    Returns:
        Lista com o texto dos elementos encontrados
    """
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        elements = soup.select(css_selector)
        
        return [elem.get_text(strip=True) for elem in elements]
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Erro ao fazer requisição: {e}")
    except Exception as e:
        raise Exception(f"Erro no web scraping: {e}")

def extract_csv_from_url(
    url: str,
    sep: str = ',',
    headers: Optional[Dict[str, str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Baixa e extrai CSV diretamente de uma URL.
    
    Args:
        url: URL do arquivo CSV
        sep: Separador do CSV
        headers: Headers HTTP para a requisição
        **kwargs: Parâmetros adicionais para pd.read_csv()
    """
    try:
        return pd.read_csv(url, sep=sep, **kwargs)
    except Exception as e:
        raise Exception(f"Erro ao baixar CSV da URL: {e}")