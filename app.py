"""Premium Streamlit dashboard for estimating student mathematics grades."""

import json
from pathlib import Path
from typing import Any

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.data_processing import TARGET_COLUMN, load_dataset

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
NO_MARKS_MODEL_PATH = ROOT / "models" / "best_model_without_previous_marks.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.json"
DATA_PATH = ROOT / "data" / "student-mat.csv"
REPORTS_PATH = ROOT / "reports"

PAGE_LABELS = {
    "Dashboard": "⌂   Dashboard",
    "Predict Grade": "◎   Predict Grade",
    "Model Analytics": "▥   Model Analytics",
    "Dataset Explorer": "▤   Dataset Explorer",
    "About Project": "ⓘ   About Project",
}

CHOICES = {
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

CHOICE_LABELS = {
    "school": {
        "GP": "Gabriel Pereira School (GP)",
        "MS": "Mousinho da Silveira School (MS)",
    },
    "sex": {"F": "Female (F)", "M": "Male (M)"},
    "address": {"U": "Urban area (U)", "R": "Rural area (R)"},
    "famsize": {
        "LE3": "3 or fewer family members (LE3)",
        "GT3": "More than 3 family members (GT3)",
    },
    "Pstatus": {
        "T": "Parents living together (T)",
        "A": "Parents living apart (A)",
    },
    "Mjob": {
        "at_home": "At-home work (at_home)",
        "health": "Health-related work (health)",
        "services": "Services (services)",
        "teacher": "Teaching (teacher)",
        "other": "Other occupation (other)",
    },
    "Fjob": {
        "at_home": "At-home work (at_home)",
        "health": "Health-related work (health)",
        "services": "Services (services)",
        "teacher": "Teaching (teacher)",
        "other": "Other occupation (other)",
    },
    "reason": {
        "course": "Course preference (course)",
        "home": "Close to home (home)",
        "reputation": "School reputation (reputation)",
        "other": "Other reason (other)",
    },
    "guardian": {
        "mother": "Mother",
        "father": "Father",
        "other": "Other guardian",
    },
    "schoolsup": {"yes": "Yes", "no": "No"},
    "famsup": {"yes": "Yes", "no": "No"},
    "paid": {"yes": "Yes", "no": "No"},
    "activities": {"yes": "Yes", "no": "No"},
    "nursery": {"yes": "Yes", "no": "No"},
    "higher": {"yes": "Yes", "no": "No"},
    "internet": {"yes": "Yes", "no": "No"},
    "romantic": {"yes": "Yes", "no": "No"},
}

NUMERIC_BOUNDS = {
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
}

FEATURE_LABELS = {
    "school": "School",
    "sex": "Sex",
    "age": "Age (years)",
    "address": "Home location",
    "famsize": "Family size",
    "Pstatus": "Parents' living arrangement",
    "Medu": "Mother's education",
    "Fedu": "Father's education",
    "Mjob": "Mother's occupation",
    "Fjob": "Father's occupation",
    "reason": "Reason for school choice",
    "guardian": "Student's guardian",
    "schoolsup": "Additional school support",
    "famsup": "Family study support",
    "paid": "Paid subject classes",
    "activities": "Extracurricular activities",
    "nursery": "Attended nursery school",
    "higher": "Plans for higher education",
    "internet": "Internet access at home",
    "romantic": "Currently in a romantic relationship",
    "traveltime": "Travel time to school (one way)",
    "studytime": "Weekly study time (category)",
    "famrel": "Family relationship quality",
    "freetime": "Free time after school",
    "goout": "Going out with friends",
    "Dalc": "Workday alcohol use",
    "Walc": "Weekend alcohol use",
    "G1": "First-period grade (0–20)",
    "G2": "Second-period grade (0–20)",
}

FIELD_HELP = {
    "school": "Choose the school attended: GP = Gabriel Pereira; MS = Mousinho da Silveira.",
    "age": "Student's age in years (15–22).",
    "sex": "The source categories are F = Female and M = Male.",
    "address": "Home location category: U = Urban and R = Rural.",
    "famsize": "LE3 means 3 or fewer family members; GT3 means more than 3.",
    "Medu": "Mother's education: 0 = none; 1 = primary; 2 = grades 5–9; 3 = secondary; 4 = higher education.",
    "Fedu": "Father's education: 0 = none; 1 = primary; 2 = grades 5–9; 3 = secondary; 4 = higher education.",
    "Pstatus": "Parents' living arrangement: T = living together; A = living apart.",
    "Mjob": "Mother's occupation category: at_home, health-related, services, teaching, or other.",
    "Fjob": "Father's occupation category: at_home, health-related, services, teaching, or other.",
    "reason": "Reported school-choice reason: course preference, close to home, school reputation, or other.",
    "guardian": "The student's recorded primary guardian: mother, father, or another guardian.",
    "traveltime": (
        "One-way travel-time category: 1 = under 15 minutes; 2 = 15–30 minutes; "
        "3 = 30 minutes–1 hour; 4 = over 1 hour."
    ),
    "studytime": (
        "A four-band category, not an exact measurement: 1 = under 2 hours per week; "
        "2 = 2–5 hours; 3 = 5–10 hours; 4 = over 10 hours."
    ),
    "failures": "Number of previous class failures recorded in the dataset (0–4).",
    "schoolsup": "Yes = receives additional educational support from school; No = no such support is recorded.",
    "famsup": "Yes = receives educational support from family; No = no such support is recorded.",
    "paid": "Yes = takes paid extra classes in this subject; No = does not.",
    "activities": "Yes = participates in activities outside regular classes; No = does not.",
    "nursery": "Yes = attended nursery school; No = did not.",
    "higher": "Yes = intends to pursue further education after school; No = does not.",
    "internet": "Yes = internet access is available at home; No = it is not.",
    "romantic": "Yes = reports currently being in a romantic relationship; No = does not.",
    "famrel": "Self-reported family relationship quality: 1 = very poor; 2 = poor; 3 = average; 4 = good; 5 = excellent.",
    "freetime": "Self-reported free time after school: 1 = very low; 2 = low; 3 = average; 4 = high; 5 = very high.",
    "goout": "Self-reported frequency of going out with friends: 1 = very low; 2 = low; 3 = average; 4 = high; 5 = very high.",
    "Dalc": "Self-reported workday alcohol-use level: 1 = very low; 2 = low; 3 = average; 4 = high; 5 = very high.",
    "Walc": "Self-reported weekend alcohol-use level: 1 = very low; 2 = low; 3 = average; 4 = high; 5 = very high.",
    "health": "Self-reported health: 1 = very poor; 2 = poor; 3 = fair; 4 = good; 5 = very good.",
    "absences": "Number of school absences recorded in the dataset (0–93); this is a count, not a rate.",
    "G1": "First-period mathematics grade on the source dataset's 0–20 scale (0 = lowest; 20 = highest).",
    "G2": "Second-period mathematics grade on the source dataset's 0–20 scale (0 = lowest; 20 = highest).",
}

FORM_GROUPS = (
    ("School and student profile", ("school", "sex", "age", "address")),
    ("Family background", ("famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian")),
    (
        "Study habits and support",
        ("traveltime", "studytime", "failures", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet"),
    ),
    (
        "Lifestyle and previous grades",
        ("romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"),
    ),
)

PRIMARY_MODE = "With previous grades (recommended)"
SECONDARY_MODE = "Without previous grades"
SUPPORT_REVIEW_THRESHOLD = 10.0


def _needs_educator_review(grade: float) -> bool:
    return grade < SUPPORT_REVIEW_THRESHOLD


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --navy: #14213d;
            --indigo: #3448a5;
            --blue: #4776e6;
            --teal: #138b8b;
            --ink: #17243b;
            --muted: #64748b;
            --line: #e4eaf3;
            --canvas: #f5f7fb;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 92% 0%, rgba(71, 118, 230, .075), transparent 31rem),
                var(--canvas);
            color: var(--ink);
        }
        [data-testid="stHeader"] { background: rgba(245, 247, 251, .86); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101c36 0%, #17294b 100%);
        }
        [data-testid="stSidebar"] * { color: #eef4ff; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 10px;
            padding: .3rem .45rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(255,255,255,.09);
        }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.15); }
        .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: var(--navy); letter-spacing: -.025em; }
        h1 { font-weight: 750; }
        [data-testid="stMetric"] {
            background: #fff;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 5px 18px rgba(20, 33, 61, .045);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
        [data-testid="stMetricValue"] { color: var(--navy); font-weight: 750; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line);
            border-radius: 16px;
            background: rgba(255,255,255,.94);
            box-shadow: 0 5px 18px rgba(20, 33, 61, .035);
        }
        [data-testid="stForm"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            background: white;
        }
        [data-testid="stForm"] label,
        [data-testid="stForm"] [data-testid="stWidgetLabel"],
        [data-testid="stForm"] [data-testid="stWidgetLabel"] p,
        [data-testid="stForm"] [data-testid="stMarkdownContainer"] p {
            color: #233451 !important;
        }
        [data-testid="stForm"] [data-testid="stWidgetLabel"] p {
            font-weight: 600;
        }
        [data-testid="stExpander"] details > summary {
            background: #edf2fb !important;
            border: 1px solid #dbe4f3 !important;
            border-radius: 10px !important;
            color: #1b2c4a !important;
        }
        [data-testid="stExpander"] details > summary:hover {
            background: #e4ecfa !important;
            color: #243f7a !important;
        }
        [data-testid="stExpander"] details > summary p,
        [data-testid="stExpander"] details > summary span,
        [data-testid="stExpander"] details > summary svg {
            color: #1b2c4a !important;
            fill: #1b2c4a !important;
        }
        div.stButton > button:not([kind="primary"]) {
            background: #fff !important;
            border: 1px solid #cbd6e6 !important;
            color: #243b65 !important;
            border-radius: 9px;
            font-weight: 600;
        }
        div.stButton > button:not([kind="primary"]):hover {
            background: #edf3ff !important;
            border-color: #93a9d4 !important;
            color: #1d3970 !important;
        }
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
            color: #183153 !important;
        }
        div.stButton > button[kind="primary"],
        div.stFormSubmitButton > button[kind="primary"] {
            background: linear-gradient(105deg, var(--indigo), var(--blue));
            border: 0;
            color: white;
            border-radius: 10px;
            font-weight: 650;
            min-height: 2.8rem;
            box-shadow: 0 5px 15px rgba(52, 72, 165, .16);
        }
        div.stButton > button[kind="primary"]:hover,
        div.stFormSubmitButton > button[kind="primary"]:hover {
            border: 0;
            color: white;
            filter: brightness(1.05);
        }
        [data-testid="stAlert"] { border-radius: 12px; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; }
        .eyebrow {
            color: var(--teal);
            font-size: .75rem;
            font-weight: 750;
            letter-spacing: .13em;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }
        .muted-copy { color: var(--muted); }
        .brand-mark {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.4rem;
            height: 2.4rem;
            border-radius: .8rem;
            background: linear-gradient(135deg, #4f76e8, #15a6a0);
            font-weight: 800;
            color: white;
            margin-right: .55rem;
        }
        .result-grade { color: var(--indigo); font-size: 2.7rem; font-weight: 800; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _load_metrics(metrics_path: str) -> dict[str, Any]:
    with Path(metrics_path).open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


@st.cache_data(show_spinner=False)
def _load_dataset(dataset_path: str) -> pd.DataFrame:
    return load_dataset(dataset_path)


@st.cache_resource(show_spinner=False)
def _load_pipeline(model_path: str, modified_time: float) -> Any:
    del modified_time
    return joblib.load(model_path)


def _heading(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="muted-copy">{description}</p>', unsafe_allow_html=True)


def _load_workspace_data() -> tuple[dict[str, Any] | None, pd.DataFrame | None]:
    try:
        metrics = _load_metrics(str(METRICS_PATH))
    except FileNotFoundError:
        metrics = None
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read the saved model metrics: {error}")
        metrics = None

    try:
        data = _load_dataset(str(DATA_PATH))
    except FileNotFoundError:
        data = None
    except (OSError, ValueError) as error:
        st.error(f"The local dataset could not be loaded or validated: {error}")
        data = None
    return metrics, data


def _selected_metrics(metrics: dict[str, Any], without_previous_marks: bool) -> tuple[str, dict[str, Any]]:
    if without_previous_marks:
        result_set = metrics["without_previous_marks"]
        return result_set["selected_model"], result_set["models"][result_set["selected_model"]]
    name = metrics["selected_model"]
    return name, metrics["models"][name]


def _metric_card(label: str, value: str, help_text: str | None = None) -> None:
    with st.container(border=True):
        st.metric(label, value, help=help_text)


def _navigate_to_prediction() -> None:
    st.session_state["active_page"] = "Predict Grade"


def _grade_distribution_chart(data: pd.DataFrame) -> alt.Chart:
    counts = (
        data[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .rename_axis("Final grade")
        .reset_index(name="Students")
    )
    return (
        alt.Chart(counts)
        .mark_bar(color="#4776e6", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Final grade:O", title="Final grade (0–20)", sort="ascending"),
            y=alt.Y("Students:Q", title="Students"),
            tooltip=[
                alt.Tooltip("Final grade:O", title="Grade"),
                alt.Tooltip("Students:Q", title="Students", format=","),
            ],
        )
        .properties(height=300)
        .interactive()
    )


def _correlation_chart(data: pd.DataFrame) -> alt.Chart:
    correlations = data.select_dtypes(include="number").corr()[TARGET_COLUMN]
    selected_features = correlations.abs().nlargest(9).index.tolist()
    matrix = (
        data[selected_features]
        .corr()
        .rename_axis("Feature")
        .reset_index()
        .melt(id_vars="Feature", var_name="Compared with", value_name="Correlation")
    )
    return (
        alt.Chart(matrix)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X("Compared with:N", title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("Feature:N", title=None, sort=selected_features),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
                title="Correlation",
            ),
            tooltip=[
                alt.Tooltip("Feature:N"),
                alt.Tooltip("Compared with:N"),
                alt.Tooltip("Correlation:Q", format=".2f"),
            ],
        )
        .properties(height=330)
    )


def _model_comparison_chart(metrics: dict[str, Any], metric_name: str = "rmse") -> alt.Chart:
    rows = []
    for label, results in (
        ("With G1/G2", metrics["models"]),
        ("Without G1/G2", metrics["without_previous_marks"]["models"]),
    ):
        for model_name, scores in results.items():
            rows.append(
                {
                    "Model": model_name,
                    "Feature set": label,
                    "Score": scores[metric_name],
                }
            )
    chart_data = pd.DataFrame(rows)
    lower_is_better = metric_name in {"rmse", "mae", "cv_rmse"}
    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=-15)),
            y=alt.Y("Score:Q", title=metric_name.upper().replace("_", " ")),
            xOffset="Feature set:N",
            color=alt.Color(
                "Feature set:N",
                scale=alt.Scale(domain=["With G1/G2", "Without G1/G2"], range=["#3448a5", "#16a0a0"]),
                title="Feature set",
            ),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("Feature set:N"),
                alt.Tooltip("Score:Q", format=".3f"),
            ],
        )
        .properties(
            height=330,
            title=f"{metric_name.upper().replace('_', ' ')} comparison "
            f"({'lower' if lower_is_better else 'higher'} is better)",
        )
        .interactive()
    )


def _render_sidebar(metrics: dict[str, Any] | None, data: pd.DataFrame | None) -> str:
    with st.sidebar:
        st.markdown(
            '<div><span class="brand-mark">SG</span><strong>Student Grade<br>Prediction</strong></div>',
            unsafe_allow_html=True,
        )
        st.caption("AI-powered academic insights")
        st.divider()
        page = st.radio(
            "Workspace",
            options=list(PAGE_LABELS),
            format_func=PAGE_LABELS.get,
            key="active_page",
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("WORKSPACE STATUS")
        if metrics is not None and MODEL_PATH.is_file():
            st.caption("●  Trained model ready")
        else:
            st.caption("○  Trained model unavailable")
        if data is not None:
            st.caption(f"●  Dataset ready · {len(data):,} records")
        else:
            st.caption("○  Dataset unavailable")
        st.divider()
        st.caption("Built with Python · Streamlit · scikit-learn")
    return page


def _render_dashboard(metrics: dict[str, Any] | None, data: pd.DataFrame | None) -> None:
    _heading(
        "STUDENT SUCCESS · MACHINE LEARNING",
        "A clearer view of student performance.",
        "Explore real academic data and estimate final mathematics grades with transparent, "
        "measured machine-learning models.",
    )
    intro, action = st.columns([3, 1])
    with intro:
        st.info(
            "Every prediction is an estimate—not a guaranteed grade or a substitute for an educator's judgment."
        )
    with action:
        st.button(
            "Predict a grade  →",
            type="primary",
            width="stretch",
            on_click=_navigate_to_prediction,
        )

    if metrics is None:
        st.warning("No saved evaluation metrics found. Train the models with `train_model.py`.")
        values = (
            ("Dataset records", f"{len(data):,}" if data is not None else "Unavailable"),
            ("Selected model", "Unavailable"),
            ("Held-out R²", "Unavailable"),
            ("Held-out RMSE", "Unavailable"),
        )
    else:
        selected_model, score = _selected_metrics(metrics, without_previous_marks=False)
        count = len(data) if data is not None else metrics.get("dataset_rows")
        values = (
            ("Dataset records", f"{count:,}" if count is not None else "Unavailable"),
            ("Selected model", selected_model),
            ("Held-out R²", f"{score['r2']:.3f}"),
            ("Held-out RMSE", f"{score['rmse']:.2f} / 20"),
        )
        if data is None:
            st.caption("Record count reflects the data used to produce the saved evaluation metrics.")
    metric_columns = st.columns(4)
    for column, (label, value) in zip(metric_columns, values):
        with column:
            _metric_card(label, value)

    if data is None:
        st.info("Dataset charts are unavailable. Add the validated UCI `student-mat.csv` to the data/ folder.")
    else:
        st.subheader("Academic snapshot")
        left, right = st.columns([1, 1.2])
        with left:
            with st.container(border=True):
                st.markdown("#### Final grade distribution")
                st.caption("Observed final mathematics grades in the UCI dataset.")
                st.altair_chart(_grade_distribution_chart(data), width="stretch")
        with right:
            with st.container(border=True):
                st.markdown("#### Feature correlation")
                st.caption("Numeric correlations; these are associations, not causes.")
                st.altair_chart(_correlation_chart(data), width="stretch")

    if metrics is not None:
        lower, upper = st.columns([1.2, 1])
        with lower:
            with st.container(border=True):
                st.markdown("#### Model performance")
                st.caption("Held-out RMSE; previous-period grades can strongly predict G3.")
                st.altair_chart(_model_comparison_chart(metrics, "rmse"), width="stretch")
        with upper:
            selected_model, score = _selected_metrics(metrics, without_previous_marks=False)
            st.markdown("#### What the results tell us")
            st.markdown(
                f"The **{selected_model}** model was selected for the primary feature set "
                f"using training-split cross-validation (CV RMSE "
                f"**{metrics['models'][selected_model]['cv_rmse']:.2f}**). "
                f"On held-out data it scored **R² {score['r2']:.3f}**, "
                f"**RMSE {score['rmse']:.2f}**, and **MAE {score['mae']:.2f}**."
            )
            st.markdown(
                "The optional model without G1/G2 scores less well, highlighting how "
                "previous-period marks can dominate a final-grade estimate."
            )
            st.caption("Negative R² indicates performance below a test-set mean baseline.")


def _field_default(feature: str, data: pd.DataFrame | None) -> int | str | None:
    if data is None:
        return None
    if feature in CHOICES:
        modes = data[feature].mode(dropna=True)
        if modes.empty:
            return None
        value = str(modes.iloc[0])
        return value if value in CHOICES[feature] else None
    value = data[feature].median()
    if pd.isna(value):
        return None
    return int(round(float(value)))


def _render_input_field(feature: str, data: pd.DataFrame | None) -> Any:
    label = FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())
    help_text = FIELD_HELP.get(feature)
    key = f"student_{feature}"
    default = _field_default(feature, data)
    if feature in CHOICES:
        return st.selectbox(
            label,
            options=CHOICES[feature],
            format_func=lambda value: CHOICE_LABELS[feature][value],
            index=CHOICES[feature].index(default) if default in CHOICES[feature] else None,
            key=key,
            help=help_text,
            placeholder="Choose an option",
        )
    minimum, maximum = NUMERIC_BOUNDS[feature]
    return st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=1,
        key=key,
        help=help_text,
        placeholder="Enter a value",
    )


def _render_prediction(metrics: dict[str, Any] | None, data: pd.DataFrame | None) -> None:
    _heading(
        "INDIVIDUAL ESTIMATE",
        "Predict a final grade.",
        "Enter the student's documented academic and background information. Inputs are used "
        "only to calculate this estimate; the app does not save the form.",
    )
    st.caption(
        "An estimate below 10/20 surfaces an optional prompt to consider human educator "
        "review. This is not a student classification; estimates at or above the threshold "
        "do not mean support is unnecessary."
    )
    if metrics is None:
        st.warning("Saved model metrics are unavailable. Train the models before requesting a prediction.")
        return

    prediction_mode = st.radio(
        "Choose the available information",
        options=[PRIMARY_MODE, SECONDARY_MODE],
        horizontal=True,
        help="The optional model excludes G1 and G2 to show estimates based on other characteristics alone.",
    )
    without_previous_marks = prediction_mode == SECONDARY_MODE
    selected_model, score = _selected_metrics(metrics, without_previous_marks)
    model_path = NO_MARKS_MODEL_PATH if without_previous_marks else MODEL_PATH
    feature_columns = (
        metrics["without_previous_marks"]["feature_columns"]
        if without_previous_marks
        else metrics["feature_columns"]
    )

    st.info(
        f"Selected model: **{selected_model}** · Held-out R² **{score['r2']:.3f}** · "
        f"RMSE **{score['rmse']:.2f}** · MAE **{score['mae']:.2f}**."
    )
    if data is not None:
        st.caption(
            "Initial values use the dataset's observed median (numeric fields) or most common "
            "category (categorical fields). Review and change them before submitting."
        )
    else:
        st.caption("Choose or enter a value for every field before submitting.")
    if without_previous_marks:
        st.caption("This model excludes G1 and G2. Its estimates are typically less precise.")

    if st.button("Clear inputs", key="clear_prediction_form"):
        for key in list(st.session_state):
            if key.startswith("student_"):
                del st.session_state[key]
        st.session_state.pop("prediction_result", None)
        st.rerun()

    submitted: dict[str, Any] = {}
    with st.form("student_prediction_form"):
        for group_name, group_features in FORM_GROUPS:
            visible_features = [name for name in group_features if name in feature_columns]
            if not visible_features:
                continue
            with st.expander(group_name, expanded=group_name == "School and student profile"):
                field_columns = st.columns(2)
                for index, feature in enumerate(visible_features):
                    with field_columns[index % 2]:
                        submitted[feature] = _render_input_field(feature, data)
        form_submitted = st.form_submit_button("Predict Final Grade", type="primary", width="stretch")

    if form_submitted:
        missing = [feature for feature in feature_columns if submitted.get(feature) is None]
        invalid = []
        for feature in feature_columns:
            value = submitted.get(feature)
            if feature in NUMERIC_BOUNDS and value is not None:
                minimum, maximum = NUMERIC_BOUNDS[feature]
                if not minimum <= value <= maximum:
                    invalid.append(feature)
            elif feature in CHOICES and value is not None and value not in CHOICES[feature]:
                invalid.append(feature)
        if missing:
            st.error("Choose or enter a value for every field before submitting.")
        elif invalid:
            st.error("Some entries are outside the documented dataset values. Review the highlighted field ranges.")
        else:
            student = pd.DataFrame(
                [{feature: submitted[feature] for feature in feature_columns}],
                columns=feature_columns,
            )
            try:
                with st.spinner("Calculating an estimate…"):
                    pipeline = _load_pipeline(
                        str(model_path.resolve()),
                        model_path.stat().st_mtime,
                    )
                    estimate = float(pipeline.predict(student)[0])
                if not pd.notna(estimate):
                    st.error("The model returned a non-finite result. Please check the entered values.")
                else:
                    bounded_estimate = min(20.0, max(0.0, estimate))
                    st.session_state["prediction_result"] = {
                        "mode": prediction_mode,
                        "model": selected_model,
                        "score": score,
                        "grade": bounded_estimate,
                        "was_bounded": bounded_estimate != estimate,
                    }
            except (OSError, ValueError, TypeError, KeyError, EOFError, ImportError) as error:
                st.error(f"Could not load or run the saved prediction pipeline: {error}")

    result = st.session_state.get("prediction_result")
    if result and result["mode"] == prediction_mode:
        st.divider()
        with st.container(border=True):
            st.markdown("#### Estimated final grade")
            st.markdown(f'<div class="result-grade">{result["grade"]:.1f} <span style="font-size:1.2rem">/ 20</span></div>', unsafe_allow_html=True)
            st.progress(result["grade"] / 20)
            st.metric("Equivalent percentage", f"{result['grade'] / 20 * 100:.1f}%")
            if result["was_bounded"]:
                st.caption("The raw model output was bounded to the valid 0–20 grade scale.")
            st.caption(
                f"{result['model']} · held-out R² {result['score']['r2']:.3f} · "
                f"RMSE {result['score']['rmse']:.2f} · MAE {result['score']['mae']:.2f}"
            )
        st.warning(
            "This is a statistical estimate based on historical associations. It is not an "
            "actual or guaranteed grade, and it should not be used as the sole basis for "
            "consequential decisions about a student."
        )
        if _needs_educator_review(result["grade"]):
            st.info(
                "Optional educator review prompt: the estimate is below 10/20. Consider "
                "reviewing the student's context, strengths, and perspective with an educator. "
                "This is not a risk label, diagnosis, or automatic decision."
            )


def _format_model_table(results: dict[str, dict[str, float]], selected_model: str) -> pd.DataFrame:
    table = pd.DataFrame(results).T
    table.insert(0, "Selected", [name == selected_model for name in table.index])
    return table.rename(
        columns={
            "cv_rmse": "CV RMSE",
            "r2": "R²",
            "rmse": "Test RMSE",
            "mae": "Test MAE",
        }
    )


def _render_model_analytics(metrics: dict[str, Any] | None) -> None:
    _heading(
        "MODEL PERFORMANCE",
        "Compare what the models learned.",
        "Every score below comes from the same held-out test split. Model selection uses "
        "cross-validation on the training split, not the held-out results.",
    )
    if metrics is None:
        st.warning("Saved evaluation metrics are unavailable. Run `train_model.py` to generate them.")
        return

    metric_choice = st.selectbox(
        "Chart metric",
        options=["Test RMSE", "Test MAE", "Test R²", "CV RMSE"],
        help="RMSE, MAE, and CV RMSE are better when lower; R² is generally better when higher.",
    )
    metric_fields = {
        "Test RMSE": "rmse",
        "Test MAE": "mae",
        "Test R²": "r2",
        "CV RMSE": "cv_rmse",
    }
    st.altair_chart(
        _model_comparison_chart(metrics, metric_fields[metric_choice]),
        width="stretch",
    )

    primary_name = metrics["selected_model"]
    secondary = metrics["without_previous_marks"]
    primary_score = metrics["models"][primary_name]
    secondary_score = secondary["models"][secondary["selected_model"]]
    first, second = st.columns(2)
    with first:
        st.subheader("With previous grades")
        st.caption(f"Selected: {primary_name} · G1 and G2 included.")
        st.dataframe(
            _format_model_table(metrics["models"], primary_name),
            width="stretch",
            column_config={
                "CV RMSE": st.column_config.NumberColumn(format="%.3f"),
                "R²": st.column_config.NumberColumn(format="%.3f"),
                "Test RMSE": st.column_config.NumberColumn(format="%.3f"),
                "Test MAE": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    with second:
        st.subheader("Without previous grades")
        st.caption(f"Selected: {secondary['selected_model']} · G1 and G2 excluded.")
        st.dataframe(
            _format_model_table(secondary["models"], secondary["selected_model"]),
            width="stretch",
            column_config={
                "CV RMSE": st.column_config.NumberColumn(format="%.3f"),
                "R²": st.column_config.NumberColumn(format="%.3f"),
                "Test RMSE": st.column_config.NumberColumn(format="%.3f"),
                "Test MAE": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    st.markdown("#### How to read these results")
    st.markdown(
        f"- **Selection criterion:** {metrics['selection_criterion']}\n"
        f"- **Primary selected model:** {primary_name} (CV RMSE {primary_score['cv_rmse']:.3f}; "
        f"held-out R² {primary_score['r2']:.3f}, RMSE {primary_score['rmse']:.3f}, "
        f"MAE {primary_score['mae']:.3f}).\n"
        f"- **Without G1/G2:** {secondary['selected_model']} "
        f"(CV RMSE {secondary_score['cv_rmse']:.3f}; held-out R² "
        f"{secondary_score['r2']:.3f}, RMSE {secondary_score['rmse']:.3f}, "
        f"MAE {secondary_score['mae']:.3f})."
    )
    st.caption(
        "Lower RMSE and MAE mean smaller average prediction errors. Higher R² is generally "
        "better; negative R² means worse than predicting the held-out mean. Previous grades "
        "are strongly correlated with G3 and can dominate the primary results."
    )


def _render_dataset_explorer(data: pd.DataFrame | None) -> None:
    _heading(
        "DATA TRANSPARENCY",
        "Explore the source dataset.",
        "Inspect validated, read-only summaries of the public UCI Student Performance "
        "mathematics dataset.",
    )
    st.markdown(
        "Source: [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance) "
        "— mathematics subject, `student-mat.csv`."
    )
    if data is None:
        st.warning("The local dataset is unavailable. Download and validate it before exploring records.")
        return

    total_missing = int(data.isna().sum().sum())
    duplicate_count = int(data.duplicated().sum())
    stats = st.columns(4)
    values = [
        ("Records", f"{len(data):,}"),
        ("Columns", f"{len(data.columns)}"),
        ("Missing values", f"{total_missing}"),
        ("Duplicate rows", f"{duplicate_count}"),
    ]
    for column, (label, value) in zip(stats, values):
        with column:
            _metric_card(label, value)

    summary_tab, schema_tab, preview_tab = st.tabs(["Grade overview", "Column information", "Data preview"])
    with summary_tab:
        chart_column, stats_column = st.columns([1.1, 1])
        with chart_column:
            st.markdown("#### Final grade distribution")
            st.altair_chart(_grade_distribution_chart(data), width="stretch")
        with stats_column:
            st.markdown("#### Numeric feature correlation")
            st.altair_chart(_correlation_chart(data), width="stretch")
        st.markdown("#### Descriptive statistics")
        st.dataframe(data.describe().T, width="stretch")
    with schema_tab:
        information = pd.DataFrame(
            {
                "Column": data.columns,
                "Data type": [str(data[column].dtype) for column in data.columns],
                "Missing": [int(data[column].isna().sum()) for column in data.columns],
                "Unique values": [int(data[column].nunique(dropna=True)) for column in data.columns],
                "Role": [
                    "Prediction target" if column == TARGET_COLUMN else "Input feature"
                    for column in data.columns
                ],
            }
        )
        st.dataframe(information, hide_index=True, width="stretch")
        st.caption("G3 is the final-grade target. It is never passed into a prediction pipeline.")
    with preview_tab:
        st.caption(
            "A small read-only preview of the public, anonymized source data. "
            "No student names or identifiers are included."
        )
        preview_columns = ["studytime", "failures", "absences", "G1", "G2", "G3"]
        st.dataframe(data[preview_columns].head(10), hide_index=True, width="stretch")
        st.caption(
            "`studytime` is a four-band weekly-study-time category (not an exact hour value); "
            "`absences` is a count, not an attendance rate."
        )
        st.caption(f"Showing 10 of {len(data):,} records; the original dataset is never modified.")


def _render_about() -> None:
    _heading(
        "PROJECT NOTES",
        "About this project.",
        "An academic machine-learning project focused on transparent grade estimation, "
        "reproducible evaluation, and responsible interpretation.",
    )
    left, right = st.columns([1.2, 1])
    with left:
        with st.container(border=True):
            st.markdown("#### Objective")
            st.write(
                "Estimate a student's final mathematics grade (G3) on the 0–20 scale using "
                "historical academic, family, and behavioral attributes from the UCI dataset."
            )
            st.markdown("#### Problem statement")
            st.write(
                "Use a reproducible regression workflow to compare several models and report "
                "their performance on held-out data. The optional no-previous-grades analysis "
                "shows how estimates change when G1 and G2 are not available."
            )
            st.markdown("#### Dataset source")
            st.write(
                "Cortez, P. (2008), Student Performance, UCI Machine Learning Repository. "
                "[Dataset page](https://archive.ics.uci.edu/dataset/320/student+performance) · "
                "[DOI 10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T)."
            )
        with st.container(border=True):
            st.markdown("#### Machine-learning workflow")
            flow = st.columns(5)
            workflow = [
                ("01", "Validate", "UCI schema"),
                ("02", "Split", "80 / 20"),
                ("03", "Preprocess", "Training only"),
                ("04", "Compare", "3 regressors"),
                ("05", "Evaluate", "Held-out set"),
            ]
            for column, (number, title, caption) in zip(flow, workflow):
                with column:
                    st.markdown(f"**{number} · {title}**")
                    st.caption(caption)
    with right:
        with st.container(border=True):
            st.markdown("#### Technology")
            st.write("Python 3.14 · Streamlit · pandas · scikit-learn · Altair · joblib")
            st.markdown("#### Algorithms")
            st.write(
                "- **Linear Regression:** linear baseline relating input features to grade.\n"
                "- **Ridge Regression:** linear regression with L2 regularization.\n"
                "- **Decision Tree Regressor:** bounded tree model for nonlinear feature splits."
            )
            st.markdown("#### Evaluation")
            st.write(
                "A seeded 80/20 split is shared by both feature sets. Pipelines fit "
                "imputation, one-hot encoding, and numeric scaling only on training data. "
                "Training-split cross-validation RMSE selects the model; held-out R², RMSE, "
                "and MAE report test performance."
            )
        with st.container(border=True):
            st.markdown("#### Limitations and responsible use")
            st.write(
                "The dataset is modest in size and covers two Portuguese schools; sample bias "
                "and limited generalization are possible. G1 and G2 are closely associated with "
                "G3, which can dominate primary-model estimates. `absences` is an absence count, "
                "not an attendance rate; `studytime` is a four-band weekly-hours category, not "
                "an exact hours value. Historical associations do not establish causation. "
                "Predictions should not be used as the sole basis for consequential decisions "
                "about students."
            )
            st.write(
                "An estimate below 10/20 prompts optional human educator review only. The "
                "threshold is not a validated risk classifier or diagnosis; consider each "
                "student's broader context, strengths, and perspective."
            )
            st.caption("Model outputs are estimates, not official or guaranteed grades.")

    with st.expander("Project structure"):
        st.code(
            "app.py                    Streamlit dashboard\n"
            "train_model.py           Training and evaluation entry point\n"
            "download_dataset.py      UCI dataset acquisition\n"
            "src/data_processing.py  Validation, loading, and exploration\n"
            "src/model_training.py   Preprocessing pipelines and evaluation\n"
            "tests/                   Automated tests\n"
            "data/                    Downloaded UCI dataset (not tracked)\n"
            "models/                  Generated pipelines and metrics",
            language="text",
        )


def main() -> None:
    st.set_page_config(
        page_title="Student Grade Prediction",
        page_icon="📘",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    metrics, data = _load_workspace_data()
    page = _render_sidebar(metrics, data)

    if page == "Dashboard":
        _render_dashboard(metrics, data)
    elif page == "Predict Grade":
        _render_prediction(metrics, data)
    elif page == "Model Analytics":
        _render_model_analytics(metrics)
    elif page == "Dataset Explorer":
        _render_dataset_explorer(data)
    else:
        _render_about()


if __name__ == "__main__":
    main()
