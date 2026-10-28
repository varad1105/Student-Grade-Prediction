"""Streamlit startup and prediction-form smoke tests."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import SUPPORT_REVIEW_THRESHOLD, _needs_educator_review


def test_educator_review_prompt_uses_strict_below_ten_threshold():
    assert _needs_educator_review(SUPPORT_REVIEW_THRESHOLD - 0.1)
    assert not _needs_educator_review(SUPPORT_REVIEW_THRESHOLD)
    assert not _needs_educator_review(SUPPORT_REVIEW_THRESHOLD + 0.1)


def test_prediction_page_renders_all_model_inputs_and_predicts():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    metrics_path = app_path.parent / "models" / "model_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Predict Grade").run(timeout=30)

    assert not app.exception
    assert len(app.selectbox) + len(app.number_input) == len(metrics["feature_columns"])
    submit = next(button for button in app.button if button.label == "Predict Final Grade")
    submit.click().run(timeout=30)
    assert not app.exception
    result = app.session_state["prediction_result"]
    assert 0 <= result["grade"] <= 20
    assert result["model"] == metrics["selected_model"]

    mode_radio = next(radio for radio in app.radio if radio.label == "Choose the available information")
    mode_radio.set_value("Without previous grades").run(timeout=30)
    assert len(app.selectbox) + len(app.number_input) == len(
        metrics["without_previous_marks"]["feature_columns"]
    )
    submit = next(button for button in app.button if button.label == "Predict Final Grade")
    submit.click().run(timeout=30)
    assert not app.exception
    result = app.session_state["prediction_result"]
    assert result["mode"] == "Without previous grades"
    assert result["model"] == metrics["without_previous_marks"]["selected_model"]
    assert 0 <= result["grade"] <= 20


def test_dashboard_and_all_navigation_pages_render():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert any("A clearer view of student performance." in title.value for title in app.title)

    cta = next(button for button in app.button if button.label == "Predict a grade  →")
    cta.click().run(timeout=30)
    assert not app.exception
    assert app.sidebar.radio[0].value == "Predict Grade"
    assert any("Predict a final grade." in title.value for title in app.title)

    for page in ("Model Analytics", "Dataset Explorer", "About Project"):
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception, f"{page} page raised an exception"
