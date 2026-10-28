"""Tests for dataset validation and end-to-end model behavior."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data_processing import EXPECTED_COLUMNS, TARGET_COLUMN, load_dataset, validate_dataset_frame
from src.model_training import MODEL_FACTORIES, _select_by_cross_validated_rmse, train_models


def sample_data(rows: int = 40) -> pd.DataFrame:
    categories = {
        "school": ["GP", "MS"],
        "sex": ["F", "M"],
        "address": ["U", "R"],
        "famsize": ["GT3", "LE3"],
        "Pstatus": ["T", "A"],
        "Mjob": ["teacher", "health", "services", "at_home", "other"],
        "Fjob": ["teacher", "health", "services", "at_home", "other"],
        "reason": ["home", "reputation", "course", "other"],
        "guardian": ["mother", "father", "other"],
        "schoolsup": ["yes", "no"],
        "famsup": ["yes", "no"],
        "paid": ["yes", "no"],
        "activities": ["yes", "no"],
        "nursery": ["yes", "no"],
        "higher": ["yes", "no"],
        "internet": ["yes", "no"],
        "romantic": ["yes", "no"],
    }
    data: dict[str, list[object]] = {}
    for column in EXPECTED_COLUMNS:
        if column in categories:
            choices = categories[column]
            data[column] = [choices[index % len(choices)] for index in range(rows)]
        elif column == TARGET_COLUMN:
            data[column] = [(index * 7) % 21 for index in range(rows)]
        else:
            minimum, maximum = {
                "age": (15, 22),
                "Medu": (0, 4),
                "Fedu": (0, 4),
                "traveltime": (1, 4),
                "studytime": (1, 4),
                "failures": (0, 4),
                "famrel": (1, 5),
                "freetime": (1, 5),
                "goout": (1, 5),
                "Dalc": (1, 5),
                "Walc": (1, 5),
                "health": (1, 5),
                "absences": (0, 20),
                "G1": (0, 20),
                "G2": (0, 20),
            }[column]
            data[column] = [minimum + index % (maximum - minimum + 1) for index in range(rows)]
    return pd.DataFrame(data, columns=EXPECTED_COLUMNS)


def test_loads_semicolon_delimited_dataset(tmp_path):
    source = sample_data(8)
    path = tmp_path / "student-mat.csv"
    source.to_csv(path, sep=";", index=False)
    loaded = load_dataset(path)
    assert loaded.shape == source.shape
    assert loaded[TARGET_COLUMN].between(0, 20).all()


def test_missing_dataset_explains_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_dataset.py"):
        load_dataset(tmp_path / "student-mat.csv")


def test_rejects_missing_expected_columns():
    with pytest.raises(ValueError, match="missing expected columns"):
        validate_dataset_frame(pd.DataFrame({"G3": [10]}))


def test_rejects_invalid_target_values():
    data = sample_data(5)
    data.loc[0, TARGET_COLUMN] = 21
    with pytest.raises(ValueError, match="between 0 and 20"):
        validate_dataset_frame(data)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("age", "not-a-number", "must contain numeric values"),
        ("studytime", 5, "between 1 and 4"),
        ("school", "UNKNOWN", "unsupported categories"),
    ],
)
def test_rejects_invalid_predictor_values(column, value, message):
    data = sample_data(5)
    if isinstance(value, str) and column in data.select_dtypes(include="number").columns:
        data[column] = data[column].astype(object)
    data.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        validate_dataset_frame(data)


def test_allows_missing_predictors_for_pipeline_imputation():
    data = sample_data(5)
    data.loc[0, "studytime"] = np.nan
    assert pd.isna(validate_dataset_frame(data).loc[0, "studytime"])


def test_cross_validation_tie_uses_declared_model_order_not_test_metrics():
    first, second, _ = MODEL_FACTORIES
    results = {
        first: {"cv_rmse": 1.0, "rmse": 99.0},
        second: {"cv_rmse": 1.0, "rmse": 1.0},
        "Decision Tree": {"cv_rmse": 2.0, "rmse": 0.0},
    }
    assert _select_by_cross_validated_rmse(results) == first


def test_training_saves_pipelines_metrics_and_consistent_features(tmp_path):
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    metrics = train_models(sample_data(), model_dir=model_dir, report_dir=report_dir, cv_folds=3)

    assert set(metrics["models"]) == {"Linear Regression", "Ridge Regression", "Decision Tree"}
    assert metrics["selected_model"] in metrics["models"]
    assert "held-out metrics do not affect selection" in metrics["selection_tie_breaker"]
    assert "G3" not in metrics["feature_columns"]
    assert {"G1", "G2"}.issubset(metrics["feature_columns"])
    assert not {"G1", "G2"}.intersection(metrics["without_previous_marks"]["feature_columns"])
    assert (model_dir / "best_model.joblib").is_file()
    assert (model_dir / "best_model_without_previous_marks.joblib").is_file()
    assert (model_dir / "model_metrics.json").is_file()
    assert (report_dir / "model_comparison.png").is_file()
    saved = json.loads((model_dir / "model_metrics.json").read_text(encoding="utf-8"))
    assert saved["feature_columns"] == metrics["feature_columns"]


def test_saved_pipeline_returns_one_finite_grade(tmp_path):
    from joblib import load

    model_dir = tmp_path / "models"
    metrics = train_models(sample_data(), model_dir=model_dir, report_dir=tmp_path / "reports", cv_folds=3)
    pipeline = load(model_dir / "best_model.joblib")
    example = sample_data(1).drop(columns=[TARGET_COLUMN])
    prediction = pipeline.predict(example[metrics["feature_columns"]])
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_training_rejects_too_few_rows(tmp_path):
    with pytest.raises(ValueError, match="At least four"):
        train_models(sample_data(3), model_dir=tmp_path / "models", report_dir=tmp_path / "reports")
