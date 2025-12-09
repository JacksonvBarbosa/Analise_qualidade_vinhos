import os
from analise_qualidade_vinhos.config.settings import RAW_DATA_PATH, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

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
        base_folder = RAW_DATA_PATH
    elif folder == "processed":
        base_folder = PROCESSED_DATA_DIR
    elif folder == "interim":
        base_folder = INTERIM_DATA_DIR
    else:
        raise ValueError("Folder must be 'raw', 'processed' or 'interim'.")

    # Retorna o caminho absoluto
    return filename_or_path if os.path.isabs(filename_or_path) else os.path.join(base_folder, filename_or_path)
