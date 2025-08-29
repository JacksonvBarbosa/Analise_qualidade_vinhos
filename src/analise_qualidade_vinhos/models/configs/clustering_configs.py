"""
Configurações de hiperparâmetros para modelos de clustering.
"""

def get_clustering_config(model_name):
    """Retorna configuração do modelo de clustering"""
    
    configs = {
        'kmeans': {
            'import_path': 'sklearn.cluster.KMeans',
            'params': {
                'random_state': 42,
                'n_clusters': 3,
                'init': 'k-means++',
                'max_iter': 300,
                'tol': 1e-4,
                'algorithm': 'lloyd',
                'n_init': 10
            }
        },
        
        'dbscan': {
            'import_path': 'sklearn.cluster.DBSCAN',
            'params': {
                'eps': 0.5,
                'min_samples': 5,
                'metric': 'euclidean',
                'algorithm': 'auto',
                'leaf_size': 30,
                'n_jobs': -1
            }
        },
        
        'gaussian_mixture': {
            'import_path': 'sklearn.mixture.GaussianMixture',
            'params': {
                'random_state': 42,
                'n_components': 3,
                'covariance_type': 'full',
                'tol': 1e-3,
                'reg_covar': 1e-6,
                'max_iter': 100,
                'n_init': 1,
                'init_params': 'kmeans'
            }
        }
    }
    
    if model_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Modelo '{model_name}' não encontrado. Disponíveis: {available}")
    
    return configs[model_name]