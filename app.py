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
X_TRAIN_PATH = BASE_DIR / "data" / "processed" / "X_train.csv"

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
        
        /* FIXED CSS: Apply Times New Roman safely without breaking layouts */
        html, body, 
        [data-testid="stApp"], 
        [data-testid="stMarkdownContainer"], 
        [data-testid="stWidgetLabel"], 
        [data-testid="stCaptionContainer"], 
        [data-testid="stAlert"], 
        [data-testid="stExpander"], 
        input, textarea, button,
        label, p, h1, h2, h3, h4, h5, h6,
        th, td {{
            font-family: "Times New Roman", Times, serif !important;
        }}
        
        [data-testid="stExpander"] summary span {{
            font-family: "Times New Roman", Times, serif !important;
        }}

        .stButton button,
        .stDownloadButton button {{
            font-family: "Times New Roman", Times, serif !important;
        }}

        table, th, td {{
            font-family: "Times New Roman", Times, serif !important;
        }}
        
        /* Hide broken material icons */
        [data-testid="stIconMaterial"] {{
            display: none !important;
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
        
        /* FIXED CSS: Apply Times New Roman safely without breaking layouts */
        html, body, 
        [data-testid="stApp"], 
        [data-testid="stMarkdownContainer"], 
        [data-testid="stWidgetLabel"], 
        [data-testid="stCaptionContainer"], 
        [data-testid="stAlert"], 
        [data-testid="stExpander"], 
        input, textarea, button,
        label, p, h1, h2, h3, h4, h5, h6,
        th, td {{
            font-family: "Times New Roman", Times, serif !important;
        }}
        
        [data-testid="stExpander"] summary span {{
            font-family: "Times New Roman", Times, serif !important;
        }}

        .stButton button,
        .stDownloadButton button {{
            font-family: "Times New Roman", Times, serif !important;
        }}

        table, th, td {{
            font-family: "Times New Roman", Times, serif !important;
        }}
        
        /* Hide broken material icons */
        [data-testid="stIconMaterial"] {{
            display: none !important;
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

# CLEAN LABEL: Removed HTML/Markdown that was causing the "arr" glitch
with st.expander("Before You Begin: Health Details"):
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
        "Predict",
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
    st.subheader("Why did the model make this prediction?")

    try:
        # Load the original training data as the SHAP reference dataset.
        if not X_TRAIN_PATH.exists():
            raise FileNotFoundError(
                f"Training data not found: {X_TRAIN_PATH}"
            )

        X_train = pd.read_csv(X_TRAIN_PATH)

        # Validate the required input columns and preserve their order.
        missing_features = set(FEATURES) - set(X_train.columns)
        if missing_features:
            raise ValueError(
                f"Missing training features: {sorted(missing_features)}"
            )

        X_train = X_train[FEATURES]

        # Apply the same fitted preprocessing to the patient and training data.
        transformed_patient = preproc.transform(input_df)
        transformed_train = preproc.transform(X_train)

        if hasattr(transformed_patient, "toarray"):
            transformed_patient = transformed_patient.toarray()

        if hasattr(transformed_train, "toarray"):
            transformed_train = transformed_train.toarray()

        transformed_patient = np.asarray(transformed_patient)
        transformed_train = np.asarray(transformed_train)

        if len(transformed_train) == 0:
            raise ValueError("X_train.csv contains no training rows.")

        # Select up to 30 training records for the SHAP background.
        rng = np.random.default_rng(42)
        background_size = min(30, len(transformed_train))
        background_indices = rng.choice(
            len(transformed_train),
            size=background_size,
            replace=False
        )
        background = transformed_train[background_indices]

        # Feature names after preprocessing (including one-hot encoded columns).
        feature_names = list(preproc.get_feature_names_out())

        if transformed_patient.shape[1] != len(feature_names):
            raise ValueError(
                "The transformed patient data and feature names do not match."
            )

        # Explain the positive-class probability shown in the prediction.
        def predict_positive_probability(data):
            data = np.asarray(data)
            return model.predict_proba(data)[:, positive_index]

        explainer = shap.Explainer(
            predict_positive_probability,
            background,
            algorithm="permutation"
        )

        max_evaluations = 10 * (2 * len(feature_names) + 1)
        shap_result = explainer(
            transformed_patient,
            max_evals=max_evaluations
        )

        shap_values = np.asarray(shap_result.values)[0]
        base_value = float(
            np.asarray(shap_result.base_values).reshape(-1)[0]
        )

        if len(feature_names) != len(shap_values):
            raise ValueError(
                "SHAP feature names and values have different lengths."
            )

        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=transformed_patient[0],
            feature_names=feature_names
        )

        # Display this patient's SHAP waterfall plot on the website.
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(
            explanation,
            show=False,
            max_display=12
        )

        shap_fig = plt.gcf()
        st.pyplot(shap_fig, clear_figure=True)

        # Keep the same plot available to the report-download code.
        shap_buffer = BytesIO()
        shap_fig.savefig(
            shap_buffer,
            format="png",
            dpi=180,
            bbox_inches="tight",
            facecolor="white"
        )
        st.session_state["shap_plot_png"] = shap_buffer.getvalue()
        plt.close(shap_fig)

        # # -------------------------------------------------
        # # 8. SIMPLE TEXT EXPLANATION
        # # -------------------------------------------------
        # st.write(
        #     "The SHAP chart explains how each patient's health details "
        #     "influenced the diabetes prediction. "
        #     "Features shown in red increase the model's prediction, "
        #     "while features shown in blue decrease it."
        # )

    except Exception as exc:
        st.warning(
            "The prediction was generated, but its SHAP explanation "
            f"could not be displayed: {exc}"
        )
