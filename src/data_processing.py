"""Dataset validation, loading, and exploratory visualizations."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATASET_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
TARGET_COLUMN = "G3"
PREVIOUS_GRADE_COLUMNS = ("G1", "G2")
NUMERIC_RANGES = {
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
    "absences": (0, 93),
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20),
}
CATEGORICAL_VALUES = {
    "school": {"GP", "MS"},
    "sex": {"F", "M"},
    "address": {"U", "R"},
    "famsize": {"GT3", "LE3"},
    "Pstatus": {"T", "A"},
    "Mjob": {"teacher", "health", "services", "at_home", "other"},
    "Fjob": {"teacher", "health", "services", "at_home", "other"},
    "reason": {"home", "reputation", "course", "other"},
    "guardian": {"mother", "father", "other"},
    "schoolsup": {"yes", "no"},
    "famsup": {"yes", "no"},
    "paid": {"yes", "no"},
    "activities": {"yes", "no"},
    "nursery": {"yes", "no"},
    "higher": {"yes", "no"},
    "internet": {"yes", "no"},
    "romantic": {"yes", "no"},
}
EXPECTED_COLUMNS = (
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
    "G3",
)


def validate_dataset_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the UCI math dataset schema and all non-missing feature values."""
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].unique().tolist()
        raise ValueError("The dataset contains duplicate columns: " + ", ".join(duplicates))

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "The dataset is missing expected columns: " + ", ".join(missing_columns)
        )
    unexpected_columns = [column for column in data.columns if column not in EXPECTED_COLUMNS]
    if unexpected_columns:
        raise ValueError(
            "The dataset contains unexpected columns: " + ", ".join(unexpected_columns)
        )
    if data.empty:
        raise ValueError("The dataset contains no student records.")

    validated = data.copy()
    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        original_values = validated[column]
        numeric_values = pd.to_numeric(original_values, errors="coerce")
        invalid_values = original_values.notna() & numeric_values.isna()
        if invalid_values.any():
            raise ValueError(f"The {column} column must contain numeric values.")
        if column == TARGET_COLUMN and numeric_values.isna().any():
            raise ValueError("The G3 target column contains missing or non-numeric grades.")
        non_missing = numeric_values.dropna()
        if not non_missing.between(minimum, maximum).all():
            raise ValueError(
                f"The {column} column must contain values between {minimum} and {maximum}."
            )
        if not (non_missing % 1 == 0).all():
            raise ValueError(f"The {column} column must contain whole-number values.")
        validated[column] = numeric_values

    for column, allowed_values in CATEGORICAL_VALUES.items():
        invalid_values = validated[column].notna() & ~validated[column].isin(allowed_values)
        if invalid_values.any():
            unexpected = sorted(validated.loc[invalid_values, column].astype(str).unique())
            raise ValueError(
                f"The {column} column contains unsupported categories: "
                + ", ".join(unexpected)
            )
    return validated


def load_dataset(path: str | Path = "data/student-mat.csv") -> pd.DataFrame:
    """Load and validate the UCI mathematics dataset (semicolon-delimited)."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run `python download_dataset.py` "
            "to download it from UCI, or download student.zip from "
            f"{DATASET_URL} and place student-mat.csv in the data/ folder."
        )
    try:
        data = pd.read_csv(dataset_path, sep=";")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not read dataset at {dataset_path}: {error}") from error
    return validate_dataset_frame(data)


def explore_dataset(data: pd.DataFrame, output_dir: str | Path = "reports") -> dict[str, Any]:
    """Print useful dataset diagnostics and save exploratory charts."""
    data = validate_dataset_frame(data)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    diagnostics: dict[str, Any] = {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "column_types": {name: str(dtype) for name, dtype in data.dtypes.items()},
        "missing_values": {name: int(count) for name, count in data.isna().sum().items()},
        "duplicate_rows": int(data.duplicated().sum()),
        "descriptive_statistics": data.describe(include="all").fillna("").to_dict(),
        "target_distribution": {
            str(grade): int(count)
            for grade, count in data[TARGET_COLUMN].value_counts().sort_index().items()
        },
    }

    print(f"Dataset shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print("\nColumn types:\n", data.dtypes.to_string())
    print("\nMissing values:\n", data.isna().sum().to_string())
    print(f"\nDuplicate rows: {diagnostics['duplicate_rows']}")
    print("\nDescriptive statistics:\n", data.describe(include="all").to_string())
    print("\nG3 target distribution:\n", data[TARGET_COLUMN].value_counts().sort_index().to_string())

    sns.set_theme(style="whitegrid")
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.histplot(data[TARGET_COLUMN], discrete=True, binwidth=1, ax=axis, color="#4472C4")
    axis.set(title="Final Mathematics Grade Distribution", xlabel="Final grade (G3, 0-20)", ylabel="Students")
    fig.tight_layout()
    fig.savefig(output_path / "grade_distribution.png", dpi=150)
    plt.close(fig)

    numeric = data.select_dtypes(include="number")
    fig, axis = plt.subplots(figsize=(11, 8))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=axis)
    axis.set_title("Numeric Feature Correlations")
    fig.tight_layout()
    fig.savefig(output_path / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    relevant = [column for column in ("G1", "G2", "studytime", "absences", "failures") if column in data]
    fig, axes = plt.subplots(1, len(relevant), figsize=(4 * len(relevant), 4))
    if len(relevant) == 1:
        axes = [axes]
    for axis, feature in zip(axes, relevant):
        sns.regplot(data=data, x=feature, y=TARGET_COLUMN, scatter_kws={"alpha": 0.45}, line_kws={"color": "#C44E52"}, ax=axis)
        axis.set_title(f"{feature} vs G3")
    fig.tight_layout()
    fig.savefig(output_path / "feature_grade_relationships.png", dpi=150)
    plt.close(fig)
    return diagnostics
