"""
Configurações de hiperparâmetros para modelos de classificação.
Modelos são importados apenas quando solicitados.
"""

def get_classification_config(model_name):
    """
    Retorna configuração do modelo de classificação.
    
    Args:
        model_name (str): Nome do modelo
        
    Returns:
        dict: Configuração com classe do modelo e hiperparâmetros
    """
    
    configs = {
        'logistic_regression': {
            'import_path': 'sklearn.linear_model.LogisticRegression',
            'params': {
                'random_state': 42,
                'max_iter': 1000,
                'solver': 'liblinear',
                'C': 1.0,
                'penalty': 'l2'
            }
        },
        
        'random_forest': {
            'import_path': 'sklearn.ensemble.RandomForestClassifier',
            'params': {
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'n_jobs': -1
            }
        },
        
        'xgboost': {
            'import_path': 'xgboost.XGBClassifier',
            'params': {
                'random_state': 42,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'n_jobs': -1
            }
        },
        
        'lightgbm': {
            'import_path': 'lightgbm.LGBMClassifier',
            'params': {
                'random_state': 42,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': -1,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0,
                'n_jobs': -1,
                'verbose': -1
            }
        },
        
        'catboost': {
            'import_path': 'catboost.CatBoostClassifier',
            'params': {
                'random_state': 42,
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'verbose': False
            }
        },
        
        # --- NOVO MODELO: Decision Tree Classifier ---
        'tree_classifier': {
            'import_path': 'sklearn.tree.DecisionTreeClassifier',
            'params': {
                'random_state': 42,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': None
            }
        }
    }
    
    if model_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Modelo '{model_name}' não encontrado. Disponíveis: {available}")
    
    return configs[model_name]


def get_available_classification_models():
    """Retorna lista de modelos de classificação disponíveis"""
    # A lista foi atualizada para incluir o novo modelo
    return ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'tree_classifier']