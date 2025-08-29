from pathlib import Path
import sys

def init_project():
    """
    Configura o ambiente do projeto para permitir imports absolutos.
    - Adiciona a raiz e 'src' no sys.path.
    - Não cria nenhuma pasta.
    """
    # Sobe até a pasta raiz (2 níveis acima deste arquivo)
    root_path = Path(__file__).resolve().parents[2]  # E-commerce/
    src_path = root_path / "src"

    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    print(f"✅ Projeto inicializado! Raiz: {root_path}")
    return root_path
