# Student Grade Prediction

A Python machine-learning project that explores the UCI Student Performance mathematics dataset, compares three regression models, and provides a Streamlit dashboard for estimating a student's final grade (`G3`, on a 0-20 scale).

## Features

- Validates and loads the semicolon-delimited UCI `student-mat.csv`.
- Reports dataset dimensions, types, missing values, duplicates, descriptive statistics, and the final-grade distribution.
- Generates grade, correlation, and feature-relationship visualizations.
- Uses train/test splitting before fitting imputers, encoders, scalers, or estimators.
- Compares Linear Regression, Ridge Regression, and Decision Tree regression pipelines.
- Selects a model using the lowest mean cross-validated RMSE on the training split; reports R², RMSE, and MAE on the held-out test split.
- Resolves exact cross-validation RMSE ties using declared model order, without consulting held-out test metrics.
- Evaluates a separate feature set without `G1` and `G2`, the previous-period grades.
- Validates dataset predictor types, ranges, and documented categories before model training or use.
- Saves trained pipelines and actual metrics for the Streamlit app.
- Provides a five-page dashboard: Dashboard, Predict Grade, Model Analytics, Dataset Explorer, and About Project.
- Presents interactive Altair charts, side-by-side held-out model comparisons, and a grade estimate form with dataset-derived median/mode defaults.
- Preserves the model's exact feature order and offers a separate prediction pipeline without previous-period grades.

## Requirements

- Python 3.14.10 was used to verify this project. The dependencies are specified in `requirements.txt`; package support depends on compatible Python wheels being available.
- Windows PowerShell, macOS, or Linux.
- Internet access for the automatic UCI dataset download (or download it manually).

## Setup on Windows PowerShell

Open PowerShell in the project folder. This workspace already has `.venv`. Use its interpreter directly so setup does not depend on PowerShell finding the `python` command:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` does not exist, create it using an installed Python interpreter. For the Python 3.14 installation used during development:

```powershell
$basePython = "$env:LocalAppData\Programs\Python\Python314\python.exe"
& $basePython -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the venv interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The VS Code workspace is configured to select `.venv\Scripts\python.exe` and to add it to new integrated terminal PATH values. Open a **new terminal** after opening the project. In a separate PowerShell window, activate `.venv` or use the full interpreter path shown above.

## Get the dataset

Run the downloader from the project root:

```powershell
.\.venv\Scripts\python.exe download_dataset.py
```

It retrieves the official UCI archive and extracts only `student-mat.csv` into `data/`, then validates its columns and target. If automatic download fails, visit the [UCI Student Performance dataset page](https://archive.ics.uci.edu/dataset/320/student+performance), download `student.zip`, and copy `student-mat.csv` into `data/`. The raw data is excluded from Git.

**Source:** Cortez, P. (2008), *Student Performance*, UCI Machine Learning Repository, [doi:10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T). Review the dataset page for its attribution and use terms.

## Train and run

From the project root (or replace `python` with `.\.venv\Scripts\python.exe` if the command is not available in your terminal):

```powershell
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The training command creates:

- `models/`: fitted preprocessing/model pipelines and `model_metrics.json`.
- `reports/`: model comparison and exploratory visualizations.

These are generated files and are excluded from Git by default. Training must complete before the prediction page can make estimates.

## Evaluation methodology

The final grade `G3` is the regression target and is never an input feature. One reproducible 80/20 split is shared by the primary and optional comparisons. Cross-validation on the training split selects each feature-set's model by mean RMSE; the held-out test set is used to report R², RMSE, and MAE for all three models.

The primary model includes `G1` and `G2`, previous-period marks that can be strong predictors of `G3`. This makes prediction easier and may dominate other features. A second, clearly labeled evaluation excludes both marks to examine prediction from other available characteristics. R² can be negative and is not reported as accuracy.

The source `absences` feature is a count of absences, not an attendance rate; the dataset does not provide the school-days denominator needed to calculate a rate. `studytime` is a four-band category for weekly study time (<2, 2–5, 5–10, and >10 hours), not a precise number of hours.

An estimate below 10/20 displays an optional prompt to consider human educator review. This user-selected threshold should be confirmed with educators before any real-world use. It is a cautious review prompt, not a validated risk classifier, diagnosis, or automatic label. An estimate at or above the threshold does not mean support is unnecessary. Any follow-up should consider the student's broader context, strengths, and perspective; predictions must not be the sole basis for consequential decisions.

### Results from the verified UCI dataset

Training was run on 395 records with an 80/20 split and random seed 42. The selected model is **Decision Tree** for the primary feature set (training cross-validation RMSE 1.901). The optional set without previous grades selected **Ridge Regression** (training cross-validation RMSE 4.584).

| Feature set | Model | Held-out R² | Held-out RMSE | Held-out MAE |
|---|---|---:|---:|---:|
| With G1/G2 | Linear Regression | 0.724 | 2.378 | 1.647 |
| With G1/G2 | Ridge Regression | 0.726 | 2.372 | 1.639 |
| With G1/G2 | Decision Tree | 0.657 | 2.653 | 1.462 |
| Without G1/G2 | Linear Regression | 0.141 | 4.196 | 3.395 |
| Without G1/G2 | Ridge Regression | 0.142 | 4.193 | 3.393 |
| Without G1/G2 | Decision Tree | -0.041 | 4.621 | 3.536 |

These are actual results from the fixed held-out split. In particular, the negative R² for the optional Decision Tree is worse than a mean-prediction baseline on this test split.

## Tests

Run tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover semicolon-delimited loading, actionable missing-data handling, column/target/predictor validation, deterministic model-selection ties, training on synthetic test-only data, saved pipeline feature consistency, prediction output, the educator-review threshold, both prediction modes, and rendering all dashboard pages. Synthetic records are only test fixtures and are not represented as the UCI dataset or used for reported model results.

## Project structure

```text
data/                  Downloaded dataset (not tracked)
models/                Trained pipelines and metrics (generated, not tracked)
reports/               Generated visualizations (not tracked)
screenshots/            Optional screenshots
src/data_processing.py Dataset checks, loading, and EDA
src/model_training.py  Preprocessing pipelines, evaluation, and persistence
tests/                  Automated tests
app.py                  Five-page Streamlit dashboard
download_dataset.py    UCI dataset downloader
train_model.py         Training entry point
requirements.txt        Python dependencies
```

## Limitations

The dataset is relatively small and represents students from two Portuguese schools. It can contain sampling bias and may not generalize to other populations. Historical correlations do not prove causes. Predictions are estimates, not guaranteed or official grades; previous marks can dominate the primary model.

## Screenshots

Add genuine screenshots of the running application to `screenshots/` after launching it. No screenshots or hosted URLs are included until actually captured or deployed.

## GitHub and Streamlit Community Cloud

1. Create a GitHub repository and upload the project files. Do not upload `.venv`, downloaded data, secrets, or caches.
2. The generated `models/` and `reports/` files are ignored. To make a Cloud deployment immediately usable, train using the UCI data and intentionally include the required generated `models/best_model.joblib` and `models/model_metrics.json` in the deployment commit. Only include these locally generated, needed artifacts; do not include the raw dataset. Alternatively, add a controlled training step to your deployment process.
3. In [Streamlit Community Cloud](https://share.streamlit.io/), deploy the repository's `app.py` and use `requirements.txt` for dependencies.
4. Verify the hosted app after deployment before sharing a demo URL. This project has not been deployed automatically.

The assignment submission needs the actual GitHub repository URL. A hosted demo URL and ZIP archive are optional only if the assignment form says so. Do not enter a made-up repository or deployment link.

## License

The project code is provided under the MIT License in `LICENSE`. The dataset is separate; refer to the UCI dataset page for its terms and citation.

## Submission notes

**Implementation remarks:** Built a five-page Streamlit dashboard to estimate final mathematics grades from the UCI Student Performance dataset (395 records). The interface includes an interactive dashboard, grade prediction form, model analytics, dataset explorer, and project overview. Dataset columns, predictor types/ranges/categories, and target values are validated before use; model preprocessing uses training-only scikit-learn pipelines. Linear Regression, Ridge Regression, and Decision Tree models are selected by cross-validated RMSE, with declared model order as the deterministic tie-breaker, and evaluated using held-out R², RMSE, and MAE. The selected primary model was Decision Tree (R² 0.657, RMSE 2.653, MAE 1.462); the optional model without G1/G2 selected Ridge Regression (R² 0.142, RMSE 4.193, MAE 3.393). Fifteen automated tests passed, including predictor validation, tie-breaking, both prediction modes, and dashboard navigation. Both saved pipelines were loaded and their feature order, finite predictions, and recorded held-out metrics verified without retraining. The prediction page offers an optional human educator review prompt below 10/20; it is not an automatic student label. Limitations include the small two-school dataset, possible population bias, non-causal correlations, and reduced predictions when previous grades are unavailable. Predictions are estimates, not guaranteed grades.
#   S t u d e n t - G r a d e - P r e d i c t i o n  
 