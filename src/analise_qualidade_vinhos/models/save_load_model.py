# Arquivo: models/save_load.py
import os
import joblib # Salva objetos grandes como matrizes Numpy e modelos de Machine Learning
from datetime import datetime

def save_model(model, path="models_storage", name="model.pkl"):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name)
    joblib.dump(model, file_path) # transforma o objeto em formato binário (pickle - .pkl) e grava no arquivo (filename)
    print(f"✅ Modelo salvo em: {file_path}")

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    model = joblib.load(path)
    print(f"📂 Modelo carregado de: {path}")
    return model

def save_model_versioned(model, path="models_storage", prefix="model"):
    os.makedirs(path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{prefix}_{timestamp}.pkl"
    file_path = os.path.join(path, file_name)
    joblib.dump(model, file_path)
    print(f"✅ Modelo salvo com versão: {file_path}")