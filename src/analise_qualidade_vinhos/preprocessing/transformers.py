import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer


# ============================================================
# CLASSES BASES DE TRANSFORMAÇÃO PADRÃO
# ============================================================

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop=None):
        self.features_to_drop = features_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        missing = set(self.features_to_drop) - set(X.columns)
        if missing:
            print(f"Atenção: {missing} não estão no DataFrame e foram ignoradas.")
        X = X.drop(columns=[c for c in self.features_to_drop if c in X.columns])
        return X


class OneHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features or []
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, X, y=None):
        if self.features:
            self.encoder.fit(X[self.features])
        return self

    def transform(self, X):
        X = X.copy()
        if not self.features:
            return X

        encoded = pd.DataFrame(
            self.encoder.transform(X[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
            index=X.index
        )
        X = pd.concat([X.drop(columns=self.features), encoded], axis=1)
        return X


class OrdinalFeature(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, categories=None):
        """
        categories: exemplo -> [['Fundamental', 'Médio', 'Superior', 'Pós', 'Doutorado']]
        """
        self.features = features or []
        self.categories = categories
        self.encoder = OrdinalEncoder(categories=self.categories)

    def fit(self, X, y=None):
        if self.features:
            self.encoder.fit(X[self.features])
        return self

    def transform(self, X):
        X = X.copy()
        if self.features:
            X[self.features] = self.encoder.transform(X[self.features])
        return X


class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features or []
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        if self.features:
            self.scaler.fit(X[self.features])
        return self

    def transform(self, X):
        X = X.copy()
        if self.features:
            X[self.features] = self.scaler.transform(X[self.features])
        return X

# Classe de balanceamento
class Oversample(BaseEstimator, TransformerMixin):
    """
    Classe reutilizável para aplicar SMOTE (Synthetic Minority Over-sampling Technique)
    e balancear o dataset em relação à variável alvo (target).

    Parâmetros
    ----------
    target_col : str
        Nome da coluna alvo (target) no DataFrame.
    sampling_strategy : str, float, dict, optional, default='minority'
        Estratégia de oversampling passada para o SMOTE.
        Ex: 'minority', 'not majority', 0.5, {'class_0': 100, 'class_1': 200}

    Exemplo de uso
    --------------
    oversampler = Oversample(target_col='label')
    df_bal = oversampler.fit_transform(df)
    """

    def __init__(self, target_col: str, sampling_strategy='minority', random_state=42):
        self.target_col = target_col
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.target_col not in X.columns:
            raise ValueError(f"A coluna alvo '{self.target_col}' não foi encontrada no DataFrame.")

        # Separa features e target
        X_features = X.drop(columns=[self.target_col])
        y_target = X[self.target_col]

        # Aplica SMOTE
        X_resampled, y_resampled = self.smote.fit_resample(X_features, y_target)

        # Recria DataFrame balanceado
        df_resampled = pd.concat(
            [pd.DataFrame(X_resampled, columns=X_features.columns),
                pd.DataFrame(y_resampled, columns=[self.target_col])],
            axis=1
        )

        return df_resampled
    

# ============================================================
# CLASSES ADICIONAIS PADRÃO (REUTILIZÁVEIS)
# ============================================================

class FillMissingValues(BaseEstimator, TransformerMixin):
    """Imputa valores nulos automaticamente."""
    def __init__(self, strategy="mean", features=None):
        self.strategy = strategy
        self.features = features
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        if self.features:
            self.imputer.fit(X[self.features])
        return self

    def transform(self, X):
        X = X.copy()
        if self.features:
            X[self.features] = self.imputer.transform(X[self.features])
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Seleciona um subconjunto específico de colunas."""
    def __init__(self, features=None):
        self.features = features or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features].copy() if self.features else X


class RenameColumns(BaseEstimator, TransformerMixin):
    """Renomeia colunas para manter padronização."""
    def __init__(self, rename_dict=None):
        self.rename_dict = rename_dict or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.rename(columns=self.rename_dict)
