"""
Configurações de hiperparâmetros para modelos de regressão.
"""

def get_regression_config(model_name):
    """Retorna configuração do modelo de regressão"""
    
    configs = {
        'linear_regression': {
            'import_path': 'sklearn.linear_model.LinearRegression',
            'params': {
                'n_jobs': -1
            }
        },
        
        'random_forest': {
            'import_path': 'sklearn.ensemble.RandomForestRegressor',
            'params': {
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 1.0,
                'bootstrap': True,
                'n_jobs': -1
            }
        },
        
        'xgboost': {
            'import_path': 'xgboost.XGBRegressor',
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
            'import_path': 'lightgbm.LGBMRegressor',
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

        'gradient_boosting': {
            'import_path': 'sklearn.ensemble.GradientBoostingRegressor',
            'params': {
                'random_state': 42,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'subsample': 1.0,
                'max_features': None,
                'verbose': 0
            }
        },

        'catboost': {
            'import_path': 'catboost.CatBoostRegressor',
            'params': {
                'random_state': 42,
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'loss_function': 'RMSE',
                'verbose': 0,
                'thread_count': -1
            }
        },

        'extra_trees': {
            'import_path': 'sklearn.ensemble.ExtraTreesRegressor',
            'params': {
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 1.0,
                'bootstrap': False,
                'n_jobs': -1,
                'verbose': 0
            }
        }

}
    
    if model_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Modelo '{model_name}' não encontrado. Disponíveis: {available}")
    
    return configs[model_name]