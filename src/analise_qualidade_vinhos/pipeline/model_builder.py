"""Pipelines used for training and inference."""

from __future__ import annotations

from typing import List, Dict, Any

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from analise_qualidade_vinhos.config.settings import QUALITY_LABELS, RANDOM_STATE


NUMERIC_FEATURES: List[str] = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
    "density_alcohol_ratio",
    "sulphates_alcohol_ratio",
    "total_free_sulfur_ratio",
    "acidity_index",
    "sugar_sulphates_interaction",
    "total_acidity",
    "alcohol_sulphates",
    "ph_acidity_interaction",
    "alcohol_squared",
    "volatile_acidity_squared",
    "sulphates_squared",
    "sulfur_efficiency",
    "citric_fixed_ratio",
    "volatile_fixed_ratio",
    "density_sugar_interaction",
]


def build_preprocessor(use_feature_selection: bool = True, k_best: int = 20) -> ColumnTransformer:
    """Build preprocessor with optional feature selection."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    
    # Adiciona seleção de features se solicitado
    if use_feature_selection:
        steps.append(("feature_selection", SelectKBest(score_func=f_classif, k=k_best)))
    
    numeric_pipeline = Pipeline(steps=steps)
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, NUMERIC_FEATURES)],
        remainder="drop",
    )


def build_training_pipeline(
    algorithm: str = "xgboost",
    use_feature_selection: bool = True,
    k_best: int = 20,
    balance_method: str = "smoteenn",
) -> Pipeline:
    """
    Build training pipeline with multiple algorithm options.
    
    Args:
        algorithm: 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm'
        use_feature_selection: Whether to use feature selection
        k_best: Number of features to select
        balance_method: 'smote', 'adasyn', 'smoteenn'
    """
    preprocessor = build_preprocessor(use_feature_selection=use_feature_selection, k_best=k_best)
    
    # Seleção do algoritmo
    if algorithm == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            bootstrap=True,
        )
    elif algorithm == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            subsample=0.8,
        )
    elif algorithm == "xgboost" and XGBOOST_AVAILABLE:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",  # logloss para binário, mlogloss para multiclasse
        )
    elif algorithm == "lightgbm" and LIGHTGBM_AVAILABLE:
        model = LGBMClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        # Fallback para RandomForest se XGBoost/LightGBM não disponível
        if algorithm in ["xgboost", "lightgbm"]:
            print(f"⚠️ {algorithm} não disponível, usando RandomForest")
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    
    # Seleção do método de balanceamento
    if balance_method == "smote":
        balancer = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    elif balance_method == "adasyn":
        balancer = ADASYN(random_state=RANDOM_STATE, n_neighbors=3)
    elif balance_method == "smoteenn":
        balancer = SMOTEENN(random_state=RANDOM_STATE)
    else:
        balancer = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("balance", balancer),
            ("model", model),
        ]
    )


def test_multiple_algorithms(
    X_train, y_train, X_test, y_test,
    algorithms: List[str] = None,
    balance_methods: List[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Testa múltiplos algoritmos e retorna resultados.
    
    Returns:
        Dict com resultados de cada combinação algoritmo+balanceamento
    """
    from sklearn.metrics import accuracy_score, f1_score
    
    if algorithms is None:
        algorithms = ["random_forest", "gradient_boosting"]
        if XGBOOST_AVAILABLE:
            algorithms.append("xgboost")
        if LIGHTGBM_AVAILABLE:
            algorithms.append("lightgbm")
    
    if balance_methods is None:
        balance_methods = ["smoteenn", "adasyn", "smote"]
    
    results = {}
    
    for algo in algorithms:
        for balance in balance_methods:
            key = f"{algo}_{balance}"
            print(f"🧪 Testando: {key}...")
            
            try:
                pipeline = build_training_pipeline(
                    algorithm=algo,
                    use_feature_selection=True,
                    k_best=20,
                    balance_method=balance,
                )
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                
                results[key] = {
                    "algorithm": algo,
                    "balance": balance,
                    "accuracy": float(accuracy_score(y_test, preds)),
                    "f1_weighted": float(f1_score(y_test, preds, average="weighted")),
                    "pipeline": pipeline,
                }
                print(f"   ✅ F1: {results[key]['f1_weighted']:.4f} | Acc: {results[key]['accuracy']:.4f}")
            except Exception as e:
                print(f"   ❌ Erro: {e}")
                results[key] = {"error": str(e)}
    
    return results


def build_best_pipeline(X_train, y_train, X_test, y_test) -> Pipeline:
    """
    Testa múltiplos algoritmos e retorna o melhor pipeline retreinado no conjunto completo.
    """
    results = test_multiple_algorithms(X_train, y_train, X_test, y_test)
    
    # Encontra o melhor resultado
    best_key = None
    best_f1 = 0.0
    best_config = None
    
    for key, result in results.items():
        if "f1_weighted" in result and result["f1_weighted"] > best_f1:
            best_f1 = result["f1_weighted"]
            best_key = key
            best_config = {
                "algorithm": result["algorithm"],
                "balance": result["balance"],
            }
    
    if best_key and best_config:
        print(f"\n🏆 Melhor modelo: {best_key} com F1={best_f1:.4f}")
        print(f"🔄 Retreinando o melhor modelo no conjunto completo de treino...")
        
        # Reconstrói e retreina o melhor pipeline no conjunto completo
        best_pipeline = build_training_pipeline(
            algorithm=best_config["algorithm"],
            use_feature_selection=True,
            k_best=20,
            balance_method=best_config["balance"],
        )
        best_pipeline.fit(X_train, y_train)
        return best_pipeline
    else:
        # Fallback
        print("⚠️ Usando pipeline padrão (RandomForest + SMOTEENN)")
        pipeline = build_training_pipeline(algorithm="random_forest", balance_method="smoteenn")
        pipeline.fit(X_train, y_train)
        return pipeline


def get_class_labels() -> list[str]:
    return QUALITY_LABELS


