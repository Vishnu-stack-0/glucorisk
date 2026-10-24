# Diabetes Risk Prediction

## Overview
A machine learning project that estimates diabetes risk based on patient health measurements and medical history. It uses an XGBoost model and a Streamlit web interface.

## Features
- Diabetes risk prediction
- Interactive Streamlit web application
- Risk probability estimation
- SHAP-based prediction explanations
- Downloadable prediction reports

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- SHAP
- Matplotlib

## Project Structure
```text
diabetes-prediction-ml-main/
├── app.py
├── data/
├── models/
├── notebook/
├── reports/
├── src/
└── README.md
```

## Installation and Execution

1. Clone the repository:

   ```bash
   git clone YOUR_REPOSITORY_URL
   cd diabetes-prediction-ml-main
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   streamlit run app.py
   ```

## Usage
1. Enter the required health measurements and medical history.
2. Click the prediction button.
3. Review the estimated diabetes risk and explanation.
4. Download the prediction report if required.

## Disclaimer
This application is intended for educational and research purposes only. It does not diagnose diabetes or replace professional medical advice or laboratory testing.

## Author
Vishnu
