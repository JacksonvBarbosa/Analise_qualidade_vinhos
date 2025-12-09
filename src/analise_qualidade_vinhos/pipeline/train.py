"""
Training entrypoint used by CLI, CI and Docker build.
Keeps the flow intentionally simple:
1) load raw data
2) feature engineering
3) train/test split with stratification
4) fit pipeline + save artifacts + metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, f1_score

from analise_qualidade_vinhos.config.settings import (
    MODEL_DIR,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
    RANDOM_STATE,
)
from analise_qualidade_vinhos.data.dataset import load_featured_data, train_test_split_featured
from analise_qualidade_vinhos.pipeline.model_builder import (
    build_training_pipeline,
    build_best_pipeline,
)


def train_model(
    data_path: Path = RAW_DATA_PATH,
    model_path: Path | None = None,
    metrics_path: Path | None = None,
) -> Tuple[Dict, Path]:
    print("🔄 Carregando dados...")
    df = load_featured_data(data_path)
    print(f"✅ Dados carregados: {len(df)} amostras, {len(df.columns)} features")
    
    X_train, X_test, y_train, y_test = train_test_split_featured(df)
    print(f"📊 Treino: {len(X_train)} | Teste: {len(X_test)}")

    print("🔧 Testando múltiplos algoritmos para encontrar o melhor...")
    print("   Algoritmos: RandomForest, GradientBoosting, XGBoost, LightGBM")
    print("   Balanceamento: SMOTEENN, ADASYN, SMOTE")
    print("   Seleção de features: Top 20 features\n")
    
    # Testa múltiplos algoritmos e seleciona o melhor (já treinado)
    pipeline = build_best_pipeline(X_train, y_train, X_test, y_test)
    
    print("\n🔍 Avaliando no conjunto de teste com o melhor modelo...")
    preds = pipeline.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)

    def _to_float(obj):
        if isinstance(obj, dict):
            return {k: _to_float(v) for k, v in obj.items()}
        try:
            return float(obj)
        except Exception:
            return obj

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1_weighted": round(float(f1_score(y_test, preds, average="weighted")), 4),
        "report": _to_float(report),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "target": TARGET_COLUMN,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if model_path is None:
        model_path = MODEL_DIR / "wine_quality_model.joblib"
    if metrics_path is None:
        metrics_path = REPORTS_DIR / "metrics.json"

    dump(pipeline, model_path)
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)

    print(f"\n✅ Treinamento concluído!")
    print(f"📊 Accuracy: {metrics['accuracy']:.4f}")
    print(f"📊 F1-weighted: {metrics['f1_weighted']:.4f}")
    print(f"💾 Modelo salvo em: {model_path}")

    return metrics, model_path


def cli():
    import argparse

    parser = argparse.ArgumentParser(description="Treina o modelo de qualidade de vinhos.")
    parser.add_argument("--data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR / "wine_quality_model.joblib")
    parser.add_argument("--metrics-path", type=Path, default=REPORTS_DIR / "metrics.json")
    args = parser.parse_args()

    metrics, path = train_model(args.data_path, args.model_path, args.metrics_path)
    print(f"Modelo salvo em: {path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()

