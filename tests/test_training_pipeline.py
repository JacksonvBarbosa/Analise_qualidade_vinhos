from pathlib import Path

from analise_qualidade_vinhos.config import settings
from analise_qualidade_vinhos.pipeline.train import train_model


def test_training_produces_model(tmp_path: Path):
    metrics, model_path = train_model(
        data_path=settings.RAW_DATA_PATH,
        model_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
    )

    assert model_path.exists()
    assert metrics["accuracy"] > 0
    assert metrics["f1_weighted"] > 0


