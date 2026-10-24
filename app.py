from matplotlib.font_manager import font_family_aliases
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
import base64
import math
import html
import json

BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "src" / "logo.png"
MODEL_PATH = BASE_DIR / "models" / "diabetes_pipeline.joblib"

FEATURES = [
    "age",
    "gender",
    "pulse_rate",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "bmi",
    "family_diabetes",
    "hypertensive",
    "family_hypertension",
    "cardiovascular_disease",
    "stroke",
]

FONT_STACK = '"Times New Roman", Times, serif'


# -------------------------------------------------
# 1. PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# -------------------------------------------------
# 2. LOADING SCREEN
# -------------------------------------------------
def show_loading_screen(logo_path: Path, duration_ms: int = 2000):
    if not logo_path.exists():
        return

    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        #splash-screen {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: linear-gradient(
                135deg, #f0f9ff 0%, #e0f2fe 50%, #f8fafc 100%
            );
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 999999;
            animation: fadeOutSplash 0.7s ease-out
                       {duration_ms}ms forwards;
        }}
        .splash-logo {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            object-fit: cover;
            box-shadow: 0 10px 30px rgba(14,165,233,0.25);
            border: 4px solid white;
            animation: logoPulse 1.4s ease-in-out infinite;
        }}
        .splash-spinner {{
            margin-top: 24px;
            width: 36px;
            height: 36px;
            border: 3.5px solid #bae6fd;
            border-top: 3.5px solid #0ea5e9;
            border-radius: 50%;
            animation: spin 0.85s linear infinite;
        }}
        .splash-text {{
            margin-top: 16px;
            font-size: 0.95rem;
            color: #0369a1;
            font-weight: 500;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        @keyframes logoPulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        @keyframes fadeOutSplash {{
            to {{
                opacity: 0;
                visibility: hidden;
                pointer-events: none;
            }}
        }}
        /* Apply Times New Roman throughout the interface */
        html, body, [class*="st-"],
        [data-testid="stApp"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stWidgetLabel"],
        [data-testid="stCaptionContainer"],
        [data-testid="stAlert"],
        [data-testid="stExpander"],
        input, textarea, button,
        label, p, span, div,
        h1, h2, h3, h4, h5, h6,
        th, td {{
            font-family: "Times New Roman", Times, serif !important;
        }}

        /* Buttons */
        .stButton button,
        .stDownloadButton button {{
            font-family: "Times New Roman", Times, serif !important;
        }}

        /* Tables */
        table, th, td {{
            font-family: "Times New Roman", Times, serif !important;
        }}
        </style>
        <div id="splash-screen">
            <img src="data:image/png;base64,{logo_b64}"
                 class="splash-logo" alt="Logo">
            <div class="splash-spinner"></div>
            <div class="splash-text">
                Loading Diabetes Risk Assessment...
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


show_loading_screen(LOGO_PATH)


# -------------------------------------------------
# 3. BACKGROUND CSS
# -------------------------------------------------
def set_background(image_path: Path, opacity: float = 0.11):
    if not image_path.exists():
        return

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(255,255,255,{1-opacity}),
                    rgba(248,250,252,{1-opacity})
                ),
                url("data:image/png;base64,{encoded}");
            background-size: cover, 220px;
            background-repeat: no-repeat;
            background-position: center;
        }}

        .stApp, .stMarkdown, .stText, p, span, label,
        .stCaption, li {{
            color: #1e293b;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: #0f172a;
            font-weight: 700;
        }}

        div[data-testid="stForm"] {{
            background-color: rgba(255,255,255,0.95);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.4rem 1.2rem;
            box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        }}

        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            color: #1e293b;
        }}

        .stFormSubmitButton > button {{
            background: linear-gradient(135deg, #0ea5e9, #0284c7);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.65rem 1.5rem;
            width: 100%;
        }}

        .stProgress > div > div > div > div {{
            background-color: #0ea5e9;
        }}

        div[data-baseweb="popover"],
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] li,
        div[data-baseweb="menu"] li {{
            background-color: white;
            color: #1e293b;
        }}

        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="menu"] li:hover {{
            background-color: #e0f2fe;
        }}
                /* Readable download buttons in light and dark themes */
        div[data-testid="stDownloadButton"] button {{
            background-color: #0f766e !important;
            color: #ffffff !important;
            border: 1px solid #0f766e !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            width: 100%;
            min-height: 44px;
        }}

        div[data-testid="stDownloadButton"] button p,
        div[data-testid="stDownloadButton"] button span,
        div[data-testid="stDownloadButton"] button svg {{
            color: #ffffff !important;
            fill: #ffffff !important;
        }}

        div[data-testid="stDownloadButton"] button:hover {{
            background-color: #115e59 !important;
            border-color: #115e59 !important;
        }}
                /* Apply Times New Roman throughout the interface */
        html, body, [class*="st-"],
        [data-testid="stApp"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stWidgetLabel"],
        [data-testid="stCaptionContainer"],
        [data-testid="stAlert"],
        [data-testid="stExpander"],
        input, textarea, button,
        label, p, span, div,
        h1, h2, h3, h4, h5, h6,
        th, td {{
            font-family: "Times New Roman", Times, serif !important;
        }}

        /* Buttons */
        .stButton button,
        .stDownloadButton button {{
            font-family: "Times New Roman", Times, serif !important;
        }}

        /* Tables */
        table, th, td {{
            font-family: "Times New Roman", Times, serif !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_background(LOGO_PATH)


# -------------------------------------------------
# 4. LOAD TRAINED MODEL
# -------------------------------------------------
@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    saved_data = joblib.load(MODEL_PATH)

    if isinstance(saved_data, dict):
        if "pipeline" not in saved_data:
            raise ValueError(
                "The saved model dictionary has no 'pipeline' entry."
            )
        loaded_pipeline = saved_data["pipeline"]
    else:
        loaded_pipeline = saved_data

    if not hasattr(loaded_pipeline, "named_steps"):
        raise TypeError(
            "The saved object is not a scikit-learn Pipeline."
        )

    required_steps = {"preprocessor", "model"}
    missing_steps = required_steps - set(
        loaded_pipeline.named_steps.keys()
    )

    if missing_steps:
        raise ValueError(
            f"Missing pipeline steps: {sorted(missing_steps)}"
        )

    expected = list(
        getattr(loaded_pipeline, "feature_names_in_", FEATURES)
    )

    if expected != FEATURES:
        raise ValueError(
            "The model's input features do not match this app.\n"
            f"Expected by app: {FEATURES}\n"
            f"Expected by model: {expected}"
        )

    return loaded_pipeline


try:
    pipeline = load_pipeline()
    preproc = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()


# -------------------------------------------------
# 5. PATIENT INPUT FORM
# -------------------------------------------------
st.title("Diabetes Risk Assessment")
st.caption(
    "Enter your details below to receive an estimate of your diabetes risk. "
)

with st.expander("Before You Begin: Understanding Your Health Details"):
    st.markdown(
        """
        | Measurement | General reference | Unit |
        |---|---|---|
        | Age | Patient's age | Years |
        | Pulse rate | Often 60–100 at rest | Beats/min |
        | Systolic BP | Often around 90–120 | mmHg |
        | Diastolic BP | Often around 60–80 | mmHg |
        | Fasting glucose | Below 5.6 is generally normal | mmol/L |
        | BMI | 18.5–24.9 is generally normal | kg/m² |

        **Medical history:** Select Yes if the patient has the condition
        or family history, and No otherwise.

        These are general reference values, not diagnostic criteria for
        an individual. Interpret measurements with a qualified clinician.
        """
    )

with st.form("patient_form"):
    # 1. Basic information
    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age", min_value=1, max_value=120, value=45
        )

    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])

    st.divider()

    # 2. Vital signs and measurements
    st.subheader("Health Measurements")

    col1, col2 = st.columns(2)

    with col1:
        pulse_rate = st.number_input(
            "Resting Pulse Rate (beats per minute)", 30, 200, 76
        )
        systolic_bp = st.number_input(
            "Systolic Blood Pressure (mmHg)", 70, 220, 130
        )
        diastolic_bp = st.number_input(
            "Diastolic Blood Pressure (mmHg)", 40, 130, 81
        )

    with col2:
        glucose = st.number_input(
            "Fasting Blood Glucose (mmol/L)",
            min_value=0.0,
            max_value=40.0,
            value=6.90,
            step=0.01,
            format="%.2f",
        )

        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=10.0,
            max_value=60.0,
            value=24.00,
            step=0.01,
            format="%.2f",
        )

    st.divider()

    # 3. Family history
    st.subheader("Family Medical History")

    col1, col2 = st.columns(2)

    with col1:
        family_diabetes = st.selectbox(
            "Does diabetes run in your family?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

    with col2:
        family_hypertension = st.selectbox(
            "Does hypertension run in your family?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

    st.divider()

    # 4. Personal medical history
    st.subheader("Personal Medical History")

    col1, col2 = st.columns(2)

    with col1:
        hypertensive = st.selectbox(
            "Have you been diagnosed with hypertension?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )
        cardiovascular_disease = st.selectbox(
            "Have you had cardiovascular disease?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

    with col2:
        stroke = st.selectbox(
            "Have you had a stroke?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

    st.divider()

    submitted = st.form_submit_button(
        "Assess Risk",
        use_container_width=True,
    )

# -------------------------------------------------
# 6. PREDICTION, SHAP AND ARCHITECTURE
# -------------------------------------------------
if submitted:
    input_df = pd.DataFrame(
        [{
            "age": age,
            "gender": gender,
            "pulse_rate": pulse_rate,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "glucose": glucose,
            "bmi": bmi,
            "family_diabetes": family_diabetes,
            "hypertensive": hypertensive,
            "family_hypertension": family_hypertension,
            "cardiovascular_disease": cardiovascular_disease,
            "stroke": stroke,
        }],
        columns=FEATURES,
    )

    try:
        probabilities = pipeline.predict_proba(input_df)[0]
        classes = list(pipeline.classes_)
        positive_index = classes.index(1) if 1 in classes else -1

        if positive_index < 0:
            raise ValueError(
                f"Model classes do not include class 1: {classes}"
            )

        prob = float(probabilities[positive_index])
        pred = int(pipeline.predict(input_df)[0])

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    eps = 1e-9
    p_clamped = min(max(prob, eps), 1 - eps)
    raw_score = math.log(p_clamped / (1 - p_clamped))

    st.divider()
    st.subheader("Your Assessment Result")

    if pred == 1:
        st.error(
            f"Higher Estimated Risk — Estimated Risk: {prob:.1%}"
        )
    else:
        st.success(
            f"Lower Estimated Risk — Estimated Risk: {prob:.1%}"
        )

    st.progress(min(max(int(prob * 100), 0), 100))
    # st.caption(
    #     "This is a model estimate, not a medical diagnosis. "
    #     "Consult a qualified healthcare professional for interpretation."
    # )

    # Clinical reference flags
    flags = []

    if glucose >= 7:
        flags.append(
            "Fasting glucose is at or above 7 mmol/L, "
            "a diabetes-range laboratory threshold."
        )
    elif glucose >= 6.1:
        flags.append(
            "Fasting glucose is elevated and warrants clinical interpretation."
        )

    if bmi >= 30:
        flags.append("BMI is in the obesity range (30 or above).")
    elif bmi >= 25:
        flags.append("BMI is in the overweight range (25–29.9).")

    if systolic_bp >= 140 or diastolic_bp >= 90:
        flags.append(
            "Blood pressure is elevated (at least 140 systolic "
            "or 90 diastolic)."
        )

    if flags:
        st.warning(
            "Values to discuss with a healthcare professional:\n\n"
            + "\n".join(f"- {flag}" for flag in flags)
        )

    # -----------------------------------------------
    # SAVE RESULTS FOR REPORT DOWNLOAD
    # -----------------------------------------------
    st.session_state["last_report"] = {
        "Prediction": (
            "Higher Estimated Risk"
            if pred == 1
            else "Lower Estimated Risk"
        ),
        "Risk Probability (%)": round(prob * 100, 2),
        "Age": age,
        "Gender": gender,
        "Pulse Rate (bpm)": pulse_rate,
        "Systolic BP (mmHg)": systolic_bp,
        "Diastolic BP (mmHg)": diastolic_bp,
        "Fasting Glucose (mmol/L)": glucose,
        "BMI": bmi,
        "Family Diabetes": "Yes" if family_diabetes else "No",
        "Hypertensive": "Yes" if hypertensive else "No",
        "Family Hypertension": (
            "Yes" if family_hypertension else "No"
        ),
        "Cardiovascular Disease": (
            "Yes" if cardiovascular_disease else "No"
        ),
        "Stroke": "Yes" if stroke else "No",
        "Clinical Flags": "; ".join(flags) if flags else "None",
        "Disclaimer": (
            "This model estimate is not a medical diagnosis. "
            "Consult a qualified healthcare professional."
        ),
    }

    # -------------------------------------------------
    # 7. SHAP EXPLANATION
    # -------------------------------------------------
    st.divider()
    st.subheader("Understanding Your Result?")

    try:
        transformed = preproc.transform(input_df)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed = np.asarray(transformed)
        feature_names = preproc.get_feature_names_out()

        explainer = shap.TreeExplainer(model)
        shap_result = explainer(transformed)

        # Handle single-output and multi-output SHAP results.
        shap_values = np.asarray(shap_result.values)
        base_values = np.asarray(shap_result.base_values)

        if shap_values.ndim == 3:
            # Typical multi-class layout: samples, features, classes.
            shap_values = shap_values[0, :, positive_index]
        elif shap_values.ndim == 2:
            shap_values = shap_values[0]
        else:
            raise ValueError(
                f"Unexpected SHAP value shape: {shap_values.shape}"
            )

        if base_values.ndim == 2:
            base_value = float(base_values[0, positive_index])
        elif base_values.ndim == 1:
            base_value = float(base_values[0])
        else:
            base_value = float(base_values)

        if len(feature_names) != len(shap_values):
            raise ValueError(
                "SHAP feature names and values have different lengths."
            )

        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=transformed[0],
            feature_names=list(feature_names),
        )

        # Create the SHAP waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(
            explanation,
            show=False,
            max_display=12,
    )

        shap_fig = plt.gcf()
        st.pyplot(shap_fig)

        # Save the same graph for downloadable PDF and HTML reports
        shap_buffer = BytesIO()
        shap_fig.savefig(
            shap_buffer,
            format="png",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )

        st.session_state["shap_plot_png"] = shap_buffer.getvalue()

        plt.close("all")

        # st.caption(
        #     "SHAP values explain how transformed model features contribute "
        #     "to the model output. They do not establish medical causation."
        # )

    except Exception as exc:
        st.warning(
            "The prediction succeeded, but the SHAP explanation "
            f"could not be displayed: {exc}"
        )

    # -------------------------------------------------
    # 8. XGBOOST ARCHITECTURE SIMULATION
    # -------------------------------------------------
    st.divider()
    st.subheader("How the Prediction Model Works")

    n_estimators = getattr(model, "n_estimators", "N/A")
    max_depth = getattr(model, "max_depth", "N/A")
    learning_rate = getattr(model, "learning_rate", "N/A")

    all_features = [
        ("Age", str(age)),
        ("Gender", gender),
        ("Pulse Rate", f"{pulse_rate} bpm"),
        ("Systolic BP", f"{systolic_bp} mmHg"),
        ("Diastolic BP", f"{diastolic_bp} mmHg"),
        ("Glucose", f"{glucose:.2f} mmol/L"),
        ("BMI", f"{bmi:.2f}"),
        ("Family Diabetes", "Yes" if family_diabetes else "No"),
        ("Hypertensive", "Yes" if hypertensive else "No"),
        ("Family Hypertension", "Yes" if family_hypertension else "No"),
        (
            "Cardiovascular Disease",
            "Yes" if cardiovascular_disease else "No",
        ),
        ("Stroke History", "Yes" if stroke else "No"),
    ]

    feature_grid_html = "".join(
        f'<div class="feat-chip"><span class="feat-name">{name}</span>'
        f'<span class="feat-val">{val}</span></div>'
        for name, val in all_features
    )

    shown_trees = (
        min(6, n_estimators)
        if isinstance(n_estimators, (int, np.integer))
        else 6
    )
    remaining_trees = (
        n_estimators - shown_trees
        if isinstance(n_estimators, (int, np.integer))
        else None
    )

    tree_icons_html = "".join(
        f"""
        <div class="tree-icon" id="tree-{i}">
            <svg viewBox="0 0 40 50" width="30" height="38">
                <polygon points="20,2 34,22 6,22" fill="#16a34a"/>
                <polygon points="20,14 32,32 8,32" fill="#22c55e"/>
                <rect x="17" y="32" width="6" height="14" fill="#78350f"/>
            </svg>
            <span>Tree {i + 1}</span>
        </div>
        """
        for i in range(int(shown_trees))
    )

    remaining_label = (
        f"⋮ +{remaining_trees} more trees"
        if remaining_trees and remaining_trees > 0
        else ""
    )

    def sx(z):
        return 10 + (z + 6) / 12 * 200

    def sy(value):
        return 130 - value * 120

    sigmoid_polyline = " ".join(
        f"{sx(-6 + (i / 40) * 12):.1f},"
        f"{sy(1 / (1 + math.exp(-(-6 + (i / 40) * 12)))):.1f}"
        for i in range(41)
    )

    marker_z = max(-6, min(6, raw_score))
    marker_x, marker_y = sx(marker_z), sy(prob)

    class_color = "#dc2626" if pred == 1 else "#16a34a"
    class_bg = "#fee2e2" if pred == 1 else "#dcfce7"
    class_text = (
        "Higher Risk — Class 1"
        if pred == 1
        else "Lower Risk — Class 0"
    )

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #f8fafc;
            color: #0f172a;
            padding: 20px;
        }}
        .title {{
            font-size: 19px;
            font-weight: 700;
            margin-bottom: 2px;
        }}
        .subtitle {{
            font-size: 12.5px;
            color: #64748b;
            margin-bottom: 12px;
        }}
        .controls {{
            margin-bottom: 16px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .sim-btn {{
            background: #0f172a;
            color: white;
            border: none;
            padding: 8px 18px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
        }}
        .sim-btn:disabled {{
            background: #94a3b8;
            cursor: not-allowed;
        }}
        .status {{
            font-size: 12.5px;
            color: #475569;
            font-family: Consolas, monospace;
        }}
        .pipeline {{
            display: flex;
            align-items: stretch;
            gap: 0;
            overflow-x: auto;
            padding-bottom: 8px;
        }}
        .layer-card {{
            min-width: 230px;
            max-width: 230px;
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(15,23,42,0.06);
            display: flex;
            flex-direction: column;
            height: 480px;
            transition: all 0.4s ease;
            opacity: 0.55;
            transform: scale(0.97);
        }}
        .layer-card.active {{
            opacity: 1;
            transform: scale(1);
            box-shadow: 0 0 0 3px rgba(59,130,246,0.35);
        }}
        .layer-card.done {{
            opacity: 1;
            transform: scale(1);
        }}
        .layer-header {{
            padding: 9px 12px;
            font-size: 11.5px;
            font-weight: 700;
            color: white;
            border-radius: 10px 10px 0 0;
        }}
        .layer-sub {{
            font-size: 10.5px;
            color: #64748b;
            padding: 6px 12px 0;
        }}
        .layer-body {{
            padding: 10px 12px;
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }}
        .layer-body.center {{
            justify-content: center;
            align-items: center;
            text-align: center;
        }}
        .connector {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-width: 92px;
            padding: 0 4px;
            opacity: 0.4;
        }}
        .connector.active {{ opacity: 1; }}
        .connector-line {{
            width: 100%;
            height: 2px;
            background: repeating-linear-gradient(
                90deg, #94a3b8 0 6px, transparent 6px 11px
            );
            position: relative;
        }}
        .connector-line::after {{
            content: "";
            position: absolute;
            right: -2px;
            top: -4px;
            border: 5px solid transparent;
            border-left-color: #64748b;
        }}
        .connector-label {{
            font-size: 10px;
            color: #475569;
            text-align: center;
            margin-top: 6px;
            line-height: 1.35;
            font-family: Consolas, monospace;
        }}
        .feat-chip {{
            display: flex;
            justify-content: space-between;
            gap: 6px;
            font-size: 10.5px;
            padding: 4px 0;
            border-bottom: 1px dashed #e2e8f0;
            opacity: 0;
            transform: translateX(-8px);
            transition: all 0.3s ease;
        }}
        .feat-chip.show {{
            opacity: 1;
            transform: translateX(0);
        }}
        .feat-name {{ color: #475569; }}
        .feat-val {{
            color: #0f172a;
            font-weight: 600;
            font-family: Consolas, monospace;
        }}
        .tree-icon {{
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            width: 33%;
            margin-bottom: 6px;
            font-size: 9.5px;
            opacity: 0.25;
            transform: scale(0.8);
            transition: all 0.35s ease;
        }}
        .tree-icon.lit {{
            opacity: 1;
            transform: scale(1);
        }}
        .tree-grid {{ display: flex; flex-wrap: wrap; }}
        .tree-remaining {{
            font-size: 10.5px;
            color: #64748b;
            margin-top: 4px;
        }}
        .hparams {{
            font-size: 10px;
            color: #64748b;
            border-top: 1px solid #e2e8f0;
            margin-top: 10px;
            padding-top: 8px;
            line-height: 1.6;
        }}
        .formula {{
            font-family: Consolas, monospace;
            font-size: 12.5px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 8px 10px;
            margin-bottom: 10px;
        }}
        .big-value {{
            font-size: 22px;
            font-weight: 700;
            color: #ea580c;
            font-family: Consolas, monospace;
            opacity: 0;
            transition: opacity 0.5s;
        }}
        .big-value.show {{ opacity: 1; }}
        .value-label {{
            font-size: 10.5px;
            color: #64748b;
            margin-top: 4px;
        }}
        #sigmoid-marker {{
            opacity: 0;
            transition: opacity 0.6s;
        }}
        #sigmoid-marker.show {{ opacity: 1; }}
        .prob-bar-track {{
            width: 100%;
            height: 10px;
            background: #e2e8f0;
            border-radius: 6px;
            overflow: hidden;
            margin: 8px 0;
        }}
        .prob-bar-fill {{
            height: 100%;
            border-radius: 6px;
            width: 0%;
            transition: width 1.2s ease-out;
        }}
        .class-badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 12.5px;
            margin-top: 10px;
            opacity: 0;
            transform: scale(0.8);
            transition: all 0.4s ease;
        }}
        .class-badge.show {{
            opacity: 1;
            transform: scale(1);
        }}
        .threshold-note {{
            font-size: 10px;
            color: #64748b;
            margin-top: 8px;
        }}
        @media (prefers-color-scheme: dark) {{
            body {{ background: #0f172a; color: #e2e8f0; }}
            .layer-card {{ background: #1e293b; border-color: #334155; }}
            .layer-sub, .value-label, .connector-label,
            .tree-remaining, .hparams, .threshold-note {{
                color: #94a3b8;
            }}
            .feat-name {{ color: #cbd5e1; }}
            .feat-val {{ color: #f1f5f9; }}
            .formula {{ background: #0f172a; color: #e2e8f0; }}
            .prob-bar-track {{ background: #334155; }}
        }}
    </style>
    </head>
    <body>
    <div class="title">How Your Diabetes Risk Estimate Is Generated</div>
    <div class="subtitle">
        Predicted Probability = {prob:.1%} |
        Predicted Class = {class_text}
    </div>

    <div class="controls">
        <button class="sim-btn" id="runBtn" onclick="runSimulation()">
            ▶ Explore the Prediction Process
        </button>
        <span class="status" id="status">Ready</span>
    </div>

    <div class="pipeline">
        <div class="layer-card" id="layer-input">
            <div class="layer-header" style="background:#0284c7;">
                INPUT LAYER
            </div>
            <div class="layer-sub">Patient feature vector (12 features)</div>
            <div class="layer-body">{feature_grid_html}</div>
        </div>

        <div class="connector" id="conn-1">
            <div class="connector-line"></div>
            <div class="connector-label">x ∈ ℝ¹²<br>(encoded)</div>
        </div>

        <div class="layer-card" id="layer-ensemble">
            <div class="layer-header" style="background:#16a34a;">
                ENSEMBLE LAYER
            </div>
            <div class="layer-sub">Gradient-boosted decision trees</div>
            <div class="layer-body">
                <div class="tree-grid">{tree_icons_html}</div>
                <div class="tree-remaining">{remaining_label}</div>
                <div class="hparams">
                    n_estimators = {n_estimators}<br>
                    max_depth = {max_depth}<br>
                    learning_rate = {learning_rate}<br>
                    Trees are combined to produce the model score.
                </div>
            </div>
        </div>

        <div class="connector" id="conn-2">
            <div class="connector-line"></div>
            <div class="connector-label">Tree outputs</div>
        </div>

        <div class="layer-card" id="layer-agg">
            <div class="layer-header" style="background:#ea580c;">
                AGGREGATION
            </div>
            <div class="layer-sub">Raw score</div>
            <div class="layer-body center">
                <div class="formula">F(x) = combined tree score</div>
                <div class="big-value" id="raw-score">z = {raw_score:.3f}</div>
                <div class="value-label">
                    Log-odds inferred from predicted probability
                </div>
            </div>
        </div>

        <div class="connector" id="conn-3">
            <div class="connector-line"></div>
            <div class="connector-label">Score → probability</div>
        </div>

        <div class="layer-card" id="layer-act">
            <div class="layer-header" style="background:#a855f7;">
                PROBABILITY
            </div>
            <div class="layer-sub">Sigmoid reference curve</div>
            <div class="layer-body center">
                <div class="formula">σ(z) = 1 / (1 + e⁻ᶻ)</div>
                <svg viewBox="0 0 220 140" width="200" height="126">
                    <line x1="10" y1="130" x2="210" y2="130"
                          stroke="#cbd5e1" stroke-width="1"/>
                    <line x1="110" y1="10" x2="110" y2="130"
                          stroke="#cbd5e1" stroke-width="1"/>
                    <polyline points="{sigmoid_polyline}" fill="none"
                              stroke="#a855f7" stroke-width="2.5"/>
                    <g id="sigmoid-marker">
                        <line x1="{marker_x:.1f}" y1="130"
                              x2="{marker_x:.1f}" y2="{marker_y:.1f}"
                              stroke="#f472b6" stroke-dasharray="3,3"/>
                        <line x1="10" y1="{marker_y:.1f}"
                              x2="{marker_x:.1f}" y2="{marker_y:.1f}"
                              stroke="#f472b6" stroke-dasharray="3,3"/>
                        <circle cx="{marker_x:.1f}" cy="{marker_y:.1f}"
                                r="4.5" fill="#f472b6"/>
                    </g>
                </svg>
                <div class="value-label">Patient probability marker</div>
            </div>
        </div>

        <div class="connector" id="conn-4">
            <div class="connector-line"></div>
            <div class="connector-label">p = {prob:.1%}</div>
        </div>

        <div class="layer-card" id="layer-out">
            <div class="layer-header" style="background:{class_color};">
                OUTPUT LAYER
            </div>
            <div class="layer-sub">Probability and predicted class</div>
            <div class="layer-body center">
                <div class="big-value" id="final-prob"
                     style="color:{class_color};">{prob:.1%}</div>
                <div class="value-label">Estimated probability</div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" id="prob-bar"
                         style="background:{class_color};"></div>
                </div>
                <div class="class-badge" id="class-badge"
                     style="background:{class_bg};color:{class_color};">
                    {class_text}
                </div>
                <div class="threshold-note">
                    Default threshold = 0.5<br>
                    p {'≥' if pred == 1 else '<'} 0.5 → Class {pred}
                </div>
            </div>
        </div>
    </div>

    <script>
    function sleep(ms) {{
        return new Promise(resolve => setTimeout(resolve, ms));
    }}

    async function runSimulation() {{
        const btn = document.getElementById('runBtn');
        const status = document.getElementById('status');
        btn.disabled = true;
        status.textContent = "Running simulation...";

        document.querySelectorAll('.layer-card').forEach(
            el => el.classList.remove('active', 'done')
        );
        document.querySelectorAll('.connector').forEach(
            el => el.classList.remove('active')
        );
        document.querySelectorAll('.feat-chip').forEach(
            el => el.classList.remove('show')
        );
        document.querySelectorAll('.tree-icon').forEach(
            el => el.classList.remove('lit')
        );
        document.getElementById('raw-score').classList.remove('show');
        document.getElementById('sigmoid-marker').classList.remove('show');
        document.getElementById('prob-bar').style.width = '0%';
        document.getElementById('class-badge').classList.remove('show');
        document.getElementById('final-prob').classList.remove('show');

        status.textContent = "1/5 Input features...";
        const inputLayer = document.getElementById('layer-input');
        inputLayer.classList.add('active');
        const chips = document.querySelectorAll('.feat-chip');

        for (let i = 0; i < chips.length; i++) {{
            chips[i].classList.add('show');
            await sleep(60);
        }}

        await sleep(400);
        inputLayer.classList.remove('active');
        inputLayer.classList.add('done');
        document.getElementById('conn-1').classList.add('active');

        status.textContent = "2/5 Evaluating trees...";
        const ensemble = document.getElementById('layer-ensemble');
        ensemble.classList.add('active');
        const trees = document.querySelectorAll('.tree-icon');

        for (let i = 0; i < trees.length; i++) {{
            trees[i].classList.add('lit');
            await sleep(180);
        }}

        ensemble.classList.remove('active');
        ensemble.classList.add('done');
        document.getElementById('conn-2').classList.add('active');

        status.textContent = "3/5 Aggregating scores...";
        const agg = document.getElementById('layer-agg');
        agg.classList.add('active');
        document.getElementById('raw-score').classList.add('show');
        await sleep(800);
        agg.classList.remove('active');
        agg.classList.add('done');
        document.getElementById('conn-3').classList.add('active');

        status.textContent = "4/5 Calculating probability...";
        const act = document.getElementById('layer-act');
        act.classList.add('active');
        document.getElementById('sigmoid-marker').classList.add('show');
        await sleep(800);
        act.classList.remove('active');
        act.classList.add('done');
        document.getElementById('conn-4').classList.add('active');

        status.textContent = "5/5 Prediction ready";
        const output = document.getElementById('layer-out');
        output.classList.add('active');
        document.getElementById('final-prob').classList.add('show');
        document.getElementById('prob-bar').style.width =
            '{prob * 100:.1f}%';

        await sleep(600);
        document.getElementById('class-badge').classList.add('show');
        await sleep(400);

        status.textContent = "Simulation complete ✓";
        btn.disabled = false;
    }}
    </script>
    </body>
    </html>
    """

    components.html(html_code, height=640, scrolling=True)
# -------------------------------------------------
# 9. Download Your Assessment Report
# IMPORTANT: This section must remain outside if submitted.
# -------------------------------------------------
st.divider()
st.subheader("📥 Download Your Assessment Report")

if "last_report" not in st.session_state:
    st.info("Run a prediction first to generate a downloadable report.")

else:
    report = st.session_state["last_report"]
    shap_png = st.session_state.get("shap_plot_png")

    report_format = st.selectbox(
        "Select a report format",
        ["PDF", "HTML", "CSV", "JSON"],
        key="report_format",
    )

    file_data = None
    file_name = ""
    mime_type = ""

    if report_format == "CSV":
        report_df = pd.DataFrame([report])
        file_data = report_df.to_csv(index=False).encode("utf-8")
        file_name = "diabetes_risk_report.csv"
        mime_type = "text/csv"

    elif report_format == "JSON":
        file_data = json.dumps(
            report,
            indent=4,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        file_name = "diabetes_risk_report.json"
        mime_type = "application/json"

    elif report_format == "HTML":
        rows = "".join(
            f"<tr><th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(str(value))}</td></tr>"
            for key, value in report.items()
        )

        if shap_png:
            shap_b64 = base64.b64encode(shap_png).decode("ascii")
            shap_section = f"""
                <section>
                    <h2>SHAP Prediction Explanation</h2>
                    <p>
                        This chart explains how the transformed model
                        features contribute to the model output.
                    </p>
                    <img
                        src="data:image/png;base64,{shap_b64}"
                        alt="SHAP waterfall explanation"
                        style="max-width:100%;height:auto;"
                    >
                </section>
            """
        else:
            shap_section = """
                <section>
                    <h2>SHAP Prediction Explanation</h2>
                    <p>
                        A SHAP chart was not available for this prediction.
                    </p>
                </section>
            """

        html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Diabetes Risk Prediction Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 30px auto;
            padding: 0 18px;
            max-width: 950px;
            background: #ffffff;
            color: #1e293b;
            line-height: 1.6;
        }}
        h1, h2 {{
            color: #176b50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #d1d5db;
            padding: 10px;
            text-align: left;
            overflow-wrap: anywhere;
        }}
        th {{
            background: #f0f5f2;
            width: 38%;
        }}
        section {{
            margin-top: 28px;
        }}
        img {{
            display: block;
            max-width: 100%;
            height: auto;
            margin: 18px auto;
        }}
        .disclaimer {{
            border-left: 4px solid #0f766e;
            background: #f0fdfa;
            padding: 12px 16px;
            margin-top: 24px;
        }}
        @media print {{
            body {{
                margin: 10px;
            }}
        }}
    </style>
</head>
<body>
    <h1>Diabetes Risk Prediction Report</h1>
    <p>Generated by the Diabetes Risk Prediction application.</p>

    <h2>Prediction and Patient Details</h2>
    <table>
        <tbody>{rows}</tbody>
    </table>

    {shap_section}

    <div class="disclaimer">
        <strong>Medical disclaimer:</strong>
        This report is informational only. It is not a medical diagnosis
        and should not replace professional medical assessment.
    </div>
</body>
</html>
"""

        file_data = html_report.encode("utf-8")
        file_name = "diabetes_risk_report.html"
        mime_type = "text/html"

    elif report_format == "PDF":
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle,
                Image as ReportLabImage,
            )

            buffer = BytesIO()

            document = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=18 * mm,
                leftMargin=18 * mm,
                topMargin=18 * mm,
                bottomMargin=18 * mm,
                title="Diabetes Risk Prediction Report",
            )

            styles = getSampleStyleSheet()

            content = [
                Paragraph(
                    "Diabetes Risk Prediction Report",
                    styles["Title"],
                ),
                Spacer(1, 12),
                Paragraph(
                    "Prediction results and submitted patient details",
                    styles["Normal"],
                ),
                Spacer(1, 12),
            ]

            # Escape values so special characters cannot break PDF paragraphs.
            table_data = [
                [
                    Paragraph("<b>Field</b>", styles["Normal"]),
                    Paragraph("<b>Value</b>", styles["Normal"]),
                ]
            ]

            for key, value in report.items():
                table_data.append(
                    [
                        Paragraph(
                            html.escape(str(key)),
                            styles["Normal"],
                        ),
                        Paragraph(
                            html.escape(str(value)),
                            styles["Normal"],
                        ),
                    ]
                )

            report_table = Table(
                table_data,
                colWidths=[55 * mm, 115 * mm],
                repeatRows=1,
            )

            report_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor("#176b50"),
                        ),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey,
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                        (
                            "BACKGROUND",
                            (0, 1),
                            (-1, -1),
                            colors.white,
                        ),
                    ]
                )
            )

            content.extend(
                [
                    report_table,
                    Spacer(1, 18),
                ]
            )

            # Embed the SHAP chart in the PDF if it was generated.
            if shap_png:
                content.extend(
                    [
                        Paragraph(
                            "SHAP Prediction Explanation",
                            styles["Heading2"],
                        ),
                        Spacer(1, 8),
                        Paragraph(
                            "The chart shows the contribution of "
                            "transformed model features to the model output. "
                            "It does not establish medical causation.",
                            styles["Normal"],
                        ),
                        Spacer(1, 10),
                    ]
                )

                shap_image_buffer = BytesIO(shap_png)
                shap_image = ReportLabImage(
                    shap_image_buffer,
                    width=170 * mm,
                    height=102 * mm,
                    kind="proportional",
                )
                content.append(shap_image)
                content.append(Spacer(1, 14))

            content.append(
                Paragraph(
                    "<b>Medical disclaimer:</b> This report is "
                    "informational only and is not a medical diagnosis. "
                    "Consult a qualified healthcare professional.",
                    styles["Normal"],
                )
            )

            document.build(content)

            file_data = buffer.getvalue()
            file_name = "diabetes_risk_report.pdf"
            mime_type = "application/pdf"

        except ImportError:
            st.warning(
                "PDF export requires ReportLab. Install it with: "
                "`pip install reportlab`"
            )

        except Exception as exc:
            st.error(f"Could not generate the PDF report: {exc}")

    if file_data is not None:
        st.download_button(
            label=f"⬇️ Download {report_format} Report",
            data=file_data,
            file_name=file_name,
            mime=mime_type,
            use_container_width=True,
            key="download_prediction_report",
        )