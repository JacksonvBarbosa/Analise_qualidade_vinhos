# modulos
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calcula métricas de MAE, MSE, R²
def evaluate_regression(y_true, y_pred):

    '''
    MAE (Erro absoluto médio) — interpretações diretas.
    MSE (Erro quadrático médio) — penaliza mais erros grandes.
    R2 (Coeficiente de determinação) — mede o quão bem o modelo explica a variância.
    '''
    mse = mean_squared_error(y_true, y_pred)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
