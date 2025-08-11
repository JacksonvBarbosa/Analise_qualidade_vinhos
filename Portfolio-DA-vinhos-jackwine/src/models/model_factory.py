"""
Factory para criar modelos com lazy loading.
"""

import importlib
from random import randint
from random import uniform
from src.models.configs.classification_configs import get_classification_config
from src.models.configs.regression_configs import get_regression_config
from src.models.configs.clustering_configs import get_clustering_config


class ModelFactory:
    """Factory para criar modelos com lazy loading"""
    
    @staticmethod # Com o método estático é possivel chamar a função sem precisar instânciar a classe ModelFactory
    def _import_class(import_path):
        """Importa classe dinamicamente"""
        try:
            module_path, class_name = import_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Erro ao importar {import_path}: {e}")
    
    @staticmethod
    def create_classification_model(model_name, custom_params=None):
        """
        Cria modelo de classificação com lazy loading.
        
        Args:
            model_name (str): Nome do modelo
            custom_params (dict, optional): Parâmetros customizados
            
        Returns:
            Modelo instanciado
        """
        config = get_classification_config(model_name) # busca import_path e params
        model_class = ModelFactory._import_class(config['import_path'])
        
        # Usar parâmetros customizados se fornecidos
        params = config['params'].copy()  # pega os parâmetros padrão
        if custom_params:
            params.update(custom_params) # substitui/insere parâmetros customizados
            
        return model_class(**params) # retorna a instância do modelo
    
    @staticmethod
    def create_regression_model(model_name, custom_params=None):
        """Cria modelo de regressão com lazy loading"""
        config = get_regression_config(model_name)
        model_class = ModelFactory._import_class(config['import_path'])
        
        params = config['params'].copy()
        if custom_params:
            params.update(custom_params)
            
        return model_class(**params)
    
    @staticmethod
    def create_clustering_model(model_name, custom_params=None):
        """Cria modelo de clustering com lazy loading"""
        config = get_clustering_config(model_name)
        model_class = ModelFactory._import_class(config['import_path'])
        
        params = config['params'].copy()
        if custom_params:
            params.update(custom_params)
            
        return model_class(**params)