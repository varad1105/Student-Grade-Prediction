"""Train, evaluate, and save the Student Grade Prediction models."""

import json

from src.data_processing import explore_dataset, load_dataset
from src.model_training import train_models


def main() -> None:
    data = load_dataset()
    explore_dataset(data)
    metrics = train_models(data)

    print("\nHeld-out test metrics (primary model family):")
    print(json.dumps(metrics["models"], indent=2))
    print(f"\nSelected primary model: {metrics['selected_model']}")
    print("\nHeld-out test metrics (without G1/G2):")
    print(json.dumps(metrics["without_previous_marks"]["models"], indent=2))
    print(
        "\nSaved pipelines and metrics to models/; visualizations are in reports/. "
        "Model selection uses training-split cross-validation RMSE."
    )


if __name__ == "__main__":
    main()
