"""
Módulo para balanceamento de dados usando diferentes técnicas
"""
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import pandas as pd
from sklearn.model_selection import train_test_split

# Funções Puros Facíl usabilidade

# Função que retorna co-variável e a variável resposta
def balance_data(X, y, method='smote', random_state=42, verbose=True):
    """
        Balancea co-variável e variável resposta

        Parametros:
        X - Co-Variável
        y - Variável Resposta
        method - método de uso (
        smote - Oversampling sintético - cria novos exemplos.
        under - Undersampling - remove exemplos da classe majoritária.
        smoteenn - Combinação de SMOTE + Edited Nearest Neighbours.)
        random_state=42 - Gera os mesmo estado para quem for utilizar  a função.
        verbose=True - mostra infomações

        Return:
        Retorna X e y para uso especifico
    """
    if verbose:
        print(f"Distribuição original: {Counter(y)}")

    if method == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif method == 'under':
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'smoteenn':
        sampler = SMOTEENN(random_state=random_state)
    else:
        raise ValueError("Método inválido: use 'smote', 'under' ou 'smoteenn'")

    X_res, y_res = sampler.fit_resample(X, y)
    
    if verbose:
        print(f"Distribuição após balanceamento: {Counter(y_res)}")

    return X_res, y_res

# Função DataFrame balanceado
def balance_dataframe(df, target_column, method='smote', random_state=42, verbose=True):
    """

    
    Parametros:
        df - Dataframe
        target_column - Variável Resposta
        method - método de uso (
        smote - Oversampling sintético - cria novos exemplos.
        under - Undersampling - remove exemplos da classe majoritária.
        smoteenn - Combinação de SMOTE + Edited Nearest Neighbours.)
        random_state=42 - Gera os mesmo estado para quem for utilizar  a função.
        verbose=True - controla a quantidade de informação que o modelo imprime 
        no console durante o seu treinamento

    Return:
    Retorna DataFrame balanceado e com mais dados

    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_balanced, y_balanced = balance_data(X, y, method=method, random_state=random_state, verbose=verbose)
    
    df_balanced = pd.DataFrame(X_balanced, columns=X.columns)
    df_balanced[target_column] = y_balanced
    
    return df_balanced

# Função que retorna os treinos e testes balanceado separadamente
def balance_with_split(df, target_column, method='smote', test_size=0.2, 
                        random_state=42, stratify=True, verbose=True):
    """
    Parametros:
        df - Dataframe
        target_column - Variável Resposta
        method - método de uso (
        smote - Oversampling sintético - cria novos exemplos.
        under - Undersampling - remove exemplos da classe majoritária.
        smoteenn - Combinação de SMOTE + Edited Nearest Neighbours.)
        test_size - Inserir valor para uso de teste 0.2 = 20%
        random_state=42 - Gera os mesmo estado para quem for utilizar  a função.
        stratify=True - garanti que a proporção de classes no seu conjunto de 
        dados original seja a mesma nos conjuntos de treino e teste.
        verbose=True - controla a quantidade de informação que o modelo imprime 
        no console durante o seu treinamento

    Return:
    Retorna todos os dados de traino e teste
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None
    )
    
    if verbose:
        print("=== DADOS DE TREINO ===")
    
    X_train_balanced, y_train_balanced = balance_data(
        X_train, y_train, method=method, random_state=random_state, verbose=verbose
    )
    
    if verbose:
        print(f"=== DADOS DE TESTE (não balanceados) ===")
        print(f"Distribuição teste: {Counter(y_test)}")
    
    return X_train_balanced, X_test, y_train_balanced, y_test