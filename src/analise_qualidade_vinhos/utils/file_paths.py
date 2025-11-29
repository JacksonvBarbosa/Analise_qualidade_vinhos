import os
from analise_qualidade_vinhos.config.settings import DATA_RAW, DATA_PROCESSED, DATA_INTERIM

def get_file_path(filename_or_path: str, folder: str = "raw") -> str:
    """
    Retorna o caminho absoluto do arquivo.
    
    Parâmetros:
    - filename_or_path: nome do arquivo ou caminho absoluto
    - folder: pasta padrão ('raw', 'processed', 'interim')
    
    Retorna:
    - Caminho absoluto do arquivo
    """
    # Escolhe a pasta base
    if folder == "raw":
        base_folder = DATA_RAW
    elif folder == "processed":
        base_folder = DATA_PROCESSED
    elif folder == "interim":
        base_folder = DATA_INTERIM
    else:
        raise ValueError("Folder must be 'raw', 'processed' or 'interim'.")

    # Retorna o caminho absoluto
    return filename_or_path if os.path.isabs(filename_or_path) else os.path.join(base_folder, filename_or_path)
