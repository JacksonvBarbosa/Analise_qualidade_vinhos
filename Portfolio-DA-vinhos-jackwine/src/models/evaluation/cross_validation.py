from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def construir_pipeline(nome_modelo, modelo):
    """Monta pipeline com scaler e modelo."""
    return Pipeline([
        ('scaler', StandardScaler()),
        (nome_modelo, modelo)
    ])

def executar_random_search(pipeline, param_grid, X_train, y_train, cv, n_iter=10, scoring='f1_weighted'):
    """Executa RandomizedSearchCV para otimização de hiperparâmetros."""
    busca = RandomizedSearchCV(
        estimator=pipeline,
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

def avaliar_cross_validation(modelo, X_train, y_train, cv, scoring='f1_weighted'):
    """Avalia modelo usando cross_val_score."""
    scores = cross_val_score(modelo.best_estimator_, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"\nValidação Cruzada ({scoring}):")
    print(f"Média: {scores.mean():.4f} | Desvio Padrão: {scores.std():.4f}\n")

def avaliar_modelo(modelo_treinado, X_test, y_test):
    """Gera métricas e matriz de confusão para modelo treinado."""
    y_pred = modelo_treinado.predict(X_test)
    print("Melhores parâmetros:", modelo_treinado.best_params_)
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()
