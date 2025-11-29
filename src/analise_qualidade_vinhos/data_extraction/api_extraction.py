import requests
import pandas as pd
from typing import Dict, Any, Optional

def extract_from_api(
    url: str,
    method: str = 'GET',
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Extrai dados de uma API REST e retorna como DataFrame.
    
    Args:
        url: URL da API
        method: Método HTTP ('GET', 'POST', etc.)
        params: Parâmetros da query string
        headers: Headers HTTP
        json_data: Dados JSON para POST/PUT
        timeout: Timeout em segundos
        normalize: Se True, usa pd.json_normalize para achatar dados aninhados
    """
    try:
        response = requests.request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            json=json_data,
            timeout=timeout
        )
        response.raise_for_status()
        
        data = response.json()
        
        if normalize:
            return pd.json_normalize(data)
        else:
            return pd.DataFrame(data)
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Erro ao fazer requisição para API: {e}")
    except ValueError as e:
        raise Exception(f"Erro ao decodificar JSON da resposta: {e}")

def extract_from_rest_api(
    base_url: str,
    endpoint: str,
    api_key: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Extrai dados de uma API REST com autenticação por API key.
    
    Args:
        base_url: URL base da API
        endpoint: Endpoint específico
        api_key: Chave da API (se necessário)
        params: Parâmetros da query
        **kwargs: Parâmetros adicionais para extract_from_api()
    """
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    headers = kwargs.get('headers', {})
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"
        # ou headers['X-API-Key'] = api_key, dependendo da API
    
    kwargs['headers'] = headers
    
    return extract_from_api(url, params=params, **kwargs)