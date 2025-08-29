from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def executar_random_search(modelo, param_grid, X_train, y_train, cv=10, n_iter=10, scoring='f1_weighted'):
    """Executa RandomizedSearchCV para otimização de hiperparâmetros."""
    busca = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=5,
        random_state=42,
        n_jobs=-1
    )
    busca.fit(X_train, y_train)
    return busca

import matplotlib.pyplot as plt

def avaliar_cross_validation(modelo, X_train, y_train, cv=10, scoring='f1_weighted'):
    """Avalia modelo usando cross_val_score com detalhes extras."""
    if hasattr(modelo, 'best_estimator_'):
        estimator = modelo.best_estimator_
    else:
        estimator = modelo

    scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    print(f"\nValidação Cruzada ({scoring}):")
    print(f"Scores por fold: {scores}")
    print(f"Média: {scores.mean():.4f}")
    print(f"Desvio Padrão: {scores.std():.4f}")
    print(f"Melhor: {scores.max():.4f}")
    print(f"Pior: {scores.min():.4f}")

    # Visualização
    plt.boxplot(scores, vert=False)
    plt.title(f"Distribuição dos Scores ({scoring})")
    plt.xlabel("Score")
    plt.show()


def avaliar_modelo(modelo_treinado, X_test, y_test):
    """Gera métricas e matriz de confusão para modelo treinado."""
    if hasattr(modelo_treinado, 'predict'):
        y_pred = modelo_treinado.predict(X_test)
    else:
        raise AttributeError("O modelo fornecido não possui método 'predict'.")

    # Exibe melhores parâmetros se houver
    if hasattr(modelo_treinado, 'best_params_'):
        print("Melhores parâmetros:", modelo_treinado.best_params_)
    else:
        print("Melhores parâmetros: não aplicável (modelo não passou por busca de hiperparâmetros)")

    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()
