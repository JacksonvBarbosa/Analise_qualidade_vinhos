# Libs
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Função de Validação Cruzada
def cross_validate_model(X, y, model, cv=5, stratified=True):

    if stratified:
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scoring_methods = ["accuracy", "f1_weighted", "recall_weighted", "precision_weighted"]

    results = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for metric in scoring_methods:
            score = cross_val_score(pipe, X, y, cv=kf, scoring=metric)
            results[metric] = score
            print(f"{metric}: {score.mean():.4f} ± {score.std():.4f}")

    return results
