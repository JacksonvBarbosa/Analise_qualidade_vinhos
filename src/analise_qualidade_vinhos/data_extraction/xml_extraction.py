import os
import pandas as pd
import xml.etree.ElementTree as ET
from analise_qualidade_vinhos.config.settings import DATA_RAW, DATA_PROCESSED, DATA_INTERIM

def extract_xml_raw(filename_or_path: str, xpath: str = None) -> pd.DataFrame:
    """
    Lê XML da pasta DATA_RAW ou de um caminho absoluto e retorna como DataFrame.
    
    Args:
        filename_or_path: Nome do arquivo ou caminho absoluto
        xpath: XPath para extrair dados específicos (opcional)
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_RAW, filename_or_path)
    
    try:
        # Tenta usar pd.read_xml (pandas >= 1.3.0)
        return pd.read_xml(filepath, xpath=xpath)
    except AttributeError:
        # Fallback para versões antigas do pandas
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        data = []
        for elem in root:
            row = {}
            for child in elem:
                row[child.tag] = child.text
            data.append(row)
        
        return pd.DataFrame(data)

def extract_xml_processed(filename_or_path: str, xpath: str = None) -> pd.DataFrame:
    """
    Lê XML da pasta DATA_PROCESSED ou de um caminho absoluto e retorna como DataFrame.
    
    Args:
        filename_or_path: Nome do arquivo ou caminho absoluto
        xpath: XPath para extrair dados específicos (opcional)
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_PROCESSED, filename_or_path)
    
    try:
        # Tenta usar pd.read_xml (pandas >= 1.3.0)
        return pd.read_xml(filepath, xpath=xpath)
    except AttributeError:
        # Fallback para versões antigas do pandas
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        data = []
        for elem in root:
            row = {}
            for child in elem:
                row[child.tag] = child.text
            data.append(row)
        
        return pd.DataFrame(data)
    
def extract_xml_interim(filename_or_path: str, xpath: str = None) -> pd.DataFrame:
    """
    Lê XML da pasta DATA_INTERIM ou de um caminho absoluto e retorna como DataFrame.
    
    Args:
        filename_or_path: Nome do arquivo ou caminho absoluto
        xpath: XPath para extrair dados específicos (opcional)
    """
    filepath = filename_or_path if os.path.isabs(filename_or_path) else os.path.join(DATA_INTERIM, filename_or_path)
    
    try:
        # Tenta usar pd.read_xml (pandas >= 1.3.0)
        return pd.read_xml(filepath, xpath=xpath)
    except AttributeError:
        # Fallback para versões antigas do pandas
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        data = []
        for elem in root:
            row = {}
            for child in elem:
                row[child.tag] = child.text
            data.append(row)
        
        return pd.DataFrame(data)