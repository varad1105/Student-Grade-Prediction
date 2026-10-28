"""Leakage-aware model training, comparison, and artifact persistence."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.data_processing import PREVIOUS_GRADE_COLUMNS, TARGET_COLUMN, validate_dataset_frame

RANDOM_STATE = 42
MODEL_FACTORIES = {
    "Linear Regression": LinearRegression,
    "Ridge Regression": lambda: Ridge(alpha=1.0),
    "Decision Tree": lambda: DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
}


def _select_by_cross_validated_rmse(results: dict[str, dict[str, Any]]) -> str:
    """Select by training-fold CV RMSE; ties follow declared model order."""
    return min(
        MODEL_FACTORIES,
        key=lambda name: results[name]["cv_rmse"],
    )


def build_pipeline(
    features: pd.DataFrame,
    estimator: Any,
    scale_numeric: bool = True,
) -> Pipeline:
    """Build a training-only imputation, encoding, and estimator pipeline."""
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def _atomic_json_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False, suffix=".tmp"
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with temporary_path.open("w", encoding="utf-8") as temporary_file:
            json.dump(payload, temporary_file, indent=2, allow_nan=False)
            temporary_file.write("\n")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_model_save(pipeline: Pipeline, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False, suffix=".tmp") as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(pipeline, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _evaluate_model_set(
    features: pd.DataFrame,
    target: pd.Series,
    train_indices: pd.Index,
    test_indices: pd.Index,
    random_state: int,
    cv_folds: int,
) -> tuple[dict[str, dict[str, Any]], str]:
    train_features, test_features = features.loc[train_indices], features.loc[test_indices]
    train_target, test_target = target.loc[train_indices], target.loc[test_indices]
    folds = min(cv_folds, len(train_features))
    if folds < 2:
        raise ValueError("At least four dataset rows are required for training and validation.")
    cross_validator = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    results: dict[str, dict[str, Any]] = {}
    pipelines: dict[str, Pipeline] = {}

    for name, estimator_factory in MODEL_FACTORIES.items():
        pipeline = build_pipeline(
            train_features,
            estimator_factory(),
            scale_numeric=name != "Decision Tree",
        )
        cv_scores = cross_val_score(
            pipeline,
            train_features,
            train_target,
            scoring="neg_root_mean_squared_error",
            cv=cross_validator,
            error_score="raise",
        )
        pipeline.fit(train_features, train_target)
        predictions = pipeline.predict(test_features)
        results[name] = {
            "cv_rmse": float(-cv_scores.mean()),
            "r2": float(r2_score(test_target, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(test_target, predictions))),
            "mae": float(mean_absolute_error(test_target, predictions)),
        }
        pipelines[name] = pipeline

    selected_name = _select_by_cross_validated_rmse(results)
    return {
        name: {**metrics, "_pipeline": pipelines[name]}
        for name, metrics in results.items()
    }, selected_name


def _save_comparison_chart(
    full_metrics: dict[str, dict[str, Any]],
    without_marks_metrics: dict[str, dict[str, Any]],
    destination: Path,
) -> None:
    model_names = list(MODEL_FACTORIES)
    x = np.arange(len(model_names))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, [full_metrics[name]["rmse"] for name in model_names], width, label="With G1/G2")
    axis.bar(
        x + width / 2,
        [without_marks_metrics[name]["rmse"] for name in model_names],
        width,
        label="Without G1/G2",
    )
    axis.set_xticks(x, model_names)
    axis.set_ylabel("Held-out RMSE (lower is better)")
    axis.set_title("Regression Model Comparison")
    axis.legend()
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def train_models(
    data: pd.DataFrame,
    model_dir: str | Path = "models",
    report_dir: str | Path = "reports",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Train and evaluate primary and no-previous-grade model families."""
    data = validate_dataset_frame(data)
    if len(data) < 4:
        raise ValueError("At least four student records are required for model training.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    target = data[TARGET_COLUMN].astype(float)
    all_features = data.drop(columns=[TARGET_COLUMN])
    no_previous_features = all_features.drop(columns=list(PREVIOUS_GRADE_COLUMNS))
    indices = pd.Index(data.index)
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    if len(test_indices) < 2 or len(train_indices) < 2:
        raise ValueError(
            "The train/test split must contain at least two records in each set so "
            "cross-validation and R² can be evaluated."
        )

    primary_results, selected_model = _evaluate_model_set(
        all_features, target, train_indices, test_indices, random_state, cv_folds
    )
    secondary_results, secondary_selected_model = _evaluate_model_set(
        no_previous_features, target, train_indices, test_indices, random_state, cv_folds
    )

    public_primary = {
        name: {key: value for key, value in metrics.items() if key != "_pipeline"}
        for name, metrics in primary_results.items()
    }
    public_secondary = {
        name: {key: value for key, value in metrics.items() if key != "_pipeline"}
        for name, metrics in secondary_results.items()
    }
    payload: dict[str, Any] = {
        "dataset_rows": int(len(data)),
        "feature_count": int(all_features.shape[1]),
        "target": TARGET_COLUMN,
        "test_size": test_size,
        "random_state": random_state,
        "selection_criterion": (
            "Lowest mean cross-validated RMSE on the training split; held-out metrics are "
            "reported separately. Exact ties are resolved by declared model order."
        ),
        "selection_tie_breaker": (
            "For equal training-split CV RMSE, select the first model in MODEL_FACTORIES "
            "declaration order; held-out metrics do not affect selection."
        ),
        "selected_model": selected_model,
        "feature_columns": all_features.columns.tolist(),
        "models": public_primary,
        "without_previous_marks": {
            "selected_model": secondary_selected_model,
            "excluded_features": list(PREVIOUS_GRADE_COLUMNS),
            "feature_columns": no_previous_features.columns.tolist(),
            "models": public_secondary,
        },
    }

    model_path = Path(model_dir)
    report_path = Path(report_dir)
    _save_comparison_chart(public_primary, public_secondary, report_path / "model_comparison.png")

    for name, metrics in primary_results.items():
        safe_name = name.lower().replace(" ", "_")
        _atomic_model_save(metrics["_pipeline"], model_path / f"{safe_name}.joblib")
    for name, metrics in secondary_results.items():
        safe_name = name.lower().replace(" ", "_")
        _atomic_model_save(
            metrics["_pipeline"], model_path / f"{safe_name}_without_previous_marks.joblib"
        )
    _atomic_model_save(
        primary_results[selected_model]["_pipeline"], model_path / "best_model.joblib"
    )
    _atomic_model_save(
        secondary_results[secondary_selected_model]["_pipeline"],
        model_path / "best_model_without_previous_marks.joblib",
    )
    _atomic_json_save(payload, model_path / "model_metrics.json")
    return payload
