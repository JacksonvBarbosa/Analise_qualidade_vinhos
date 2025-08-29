# Libs
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

def plot_roc_curve(model, X_train, y_train, X_test, y_test):
    """
    Plota curvas ROC para treino e teste com AUC.
    Funciona para classificação binária e multiclasse (One-vs-Rest).

    Args:
        model: Modelo treinado (precisa ter predict_proba ou decision_function)
        X_train, y_train: Dados de treino
        X_test, y_test: Dados de teste
    """
    # Verifica se modelo pode gerar probabilidades
    if hasattr(model, "predict_proba"):
        y_score_train = model.predict_proba(X_train)
        y_score_test = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_score_train = model.decision_function(X_train)
        y_score_test = model.decision_function(X_test)
    else:
        raise ValueError("O modelo não suporta 'predict_proba' ou 'decision_function'.")

    classes = np.unique(y_train)

    # Caso binário, força formato [n_samples, 2]
    if len(classes) == 2:
        fpr_train, tpr_train, _ = roc_curve(y_train, y_score_train[:, 1])
        fpr_test, tpr_test, _ = roc_curve(y_test, y_score_test[:, 1])
        auc_train = auc(fpr_train, tpr_train)
        auc_test = auc(fpr_test, tpr_test)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr_train, tpr_train, label=f'Treino (AUC = {auc_train:.3f})')
        plt.plot(fpr_test, tpr_test, label=f'Teste (AUC = {auc_test:.3f})', linestyle='--')
        plt.plot([0, 1], [0, 1], 'k--', lw=0.8)
        plt.xlabel("Falso Positivo")
        plt.ylabel("Verdadeiro Positivo")
        plt.title("Curva ROC")
        plt.legend(loc="lower right")
        plt.show()

    else:
        # Multiclasse: One-vs-Rest
        y_train_bin = label_binarize(y_train, classes=classes)
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = y_train_bin.shape[1]

        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            fpr_train, tpr_train, _ = roc_curve(y_train_bin[:, i], y_score_train[:, i])
            fpr_test, tpr_test, _ = roc_curve(y_test_bin[:, i], y_score_test[:, i])
            auc_train = auc(fpr_train, tpr_train)
            auc_test = auc(fpr_test, tpr_test)

            plt.plot(fpr_train, tpr_train, label=f'Treino Classe {classes[i]} (AUC = {auc_train:.3f})')
            plt.plot(fpr_test, tpr_test, linestyle='--', 
                    label=f'Teste Classe {classes[i]} (AUC = {auc_test:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
        plt.xlabel("Falso Positivo")
        plt.ylabel("Verdadeiro Positivo")
        plt.title("Curva ROC (One-vs-Rest)")
        plt.legend(loc="lower right")
        plt.show()
