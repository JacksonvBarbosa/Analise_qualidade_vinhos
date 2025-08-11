from scipy.stats import randint, uniform

def get_classification_param_distributions(model_name):
    """Retorna distribuições de hiperparâmetros para busca aleatória."""
    param_dists = {
        'logistic_regression': {
            'C': uniform(0.01, 10),
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['liblinear', 'saga']
        },
        'random_forest': {
            'n_estimators': randint(100, 200),
            'max_depth': randint(5, 15),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 10),
            'class_weight': ['balanced', None]
        },
        'xgboost': {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 12),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.5, 1.0),
            'colsample_bytree': uniform(0.5, 1.0)
        },
        'lightgbm': {
            'n_estimators': randint(50, 300),
            'max_depth': randint(-1, 12),
            'learning_rate': uniform(0.01, 0.3),
            'num_leaves': randint(20, 60),
            'subsample': uniform(0.5, 1.0)
        },
        'catboost': {
            'iterations': randint(50, 300),
            'depth': randint(4, 10),
            'learning_rate': uniform(0.01, 0.3),
            'l2_leaf_reg': uniform(1, 10)
        },
        'tree_classifier': {
            'max_depth': randint(3, 15),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5)
        },
        'svm_classifier': {
            'C': uniform(0.1, 10),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': randint(2, 5),
            'gamma': ['scale', 'auto']
        }
    }

    if model_name not in param_dists:
        raise ValueError(f"Nenhum grid definido para o modelo '{model_name}'")
    return param_dists[model_name]
