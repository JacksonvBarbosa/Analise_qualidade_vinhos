import sys
from pathlib import Path

# Certifique se a pasta (src) do projeto is na sys.path para o teste de descorberta do CI
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
