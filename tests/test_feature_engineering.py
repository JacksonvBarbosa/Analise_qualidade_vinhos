import pandas as pd

from analise_qualidade_vinhos.features.engineering import (
    TARGET_LABELS,
    build_feature_matrix,
    bucket_quality,
    rename_columns,
)


def test_bucket_quality_labels():
    assert bucket_quality(4) == TARGET_LABELS["low"]
    assert bucket_quality(6) == TARGET_LABELS["medium"]
    assert bucket_quality(7) == TARGET_LABELS["high"]


def test_build_feature_matrix_creates_target_and_features():
    sample = pd.DataFrame(
        {
            "fixed acidity": [7.4],
            "volatile acidity": [0.7],
            "citric acid": [0.0],
            "residual sugar": [1.9],
            "chlorides": [0.076],
            "free sulfur dioxide": [11.0],
            "total sulfur dioxide": [34.0],
            "density": [0.9978],
            "pH": [3.51],
            "sulphates": [0.56],
            "alcohol": [9.4],
            "quality": [5],
        }
    )

    featured = build_feature_matrix(sample)
    expected_cols = {
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "quality_label",
    }
    assert expected_cols.issubset(featured.columns)
    assert featured.loc[0, "quality_label"] == TARGET_LABELS["low"]


