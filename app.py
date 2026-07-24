"""
Student Health Risk Predictor — Streamlit Frontend
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from inference import load_models, predict

MODEL_PATH = APP_DIR / "models" / "trained_models.pkl"

st.set_page_config(
    page_title="Student Health Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1rem; color: #555; margin-bottom: 2rem; }
    .metric-card { padding: 1.5rem; border-radius: 12px; text-align: center; }
    .metric-card h3 { margin: 0; font-size: 2rem; }
    .metric-card p { margin: 0; font-size: 0.9rem; }
    .risk-at-risk { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .risk-unhealthy { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #333; }
    .risk-fit { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: #333; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return load_models(str(MODEL_PATH))


def single_prediction(model_data):
    st.subheader("Enter Student Health Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.0, 0.1)
        heart_rate = st.slider("Heart Rate (bpm)", 40.0, 120.0, 72.0, 0.1)
        bmi = st.slider("BMI", 15.0, 45.0, 24.0, 0.1)

    with col2:
        calorie_expenditure = st.slider("Calorie Expenditure (kcal)", 500.0, 5000.0, 2200.0, 50.0)
        step_count = st.slider("Step Count", 0.0, 25000.0, 7500.0, 100.0)
        exercise_duration = st.slider("Exercise Duration (min)", 0.0, 120.0, 35.0, 1.0)

    with col3:
        water_intake = st.slider("Water Intake (liters)", 0.5, 5.0, 2.2, 0.1)
        diet_type = st.selectbox("Diet Type", ["veg", "non-veg", "vegan"])
        stress_level = st.selectbox("Stress Level", ["low", "medium", "high"])

    col4, col5 = st.columns(2)
    with col4:
        sleep_quality = st.selectbox("Sleep Quality", ["poor", "average", "good"])
        physical_activity_level = st.selectbox("Physical Activity Level", ["sedentary", "moderate", "active"])

    with col5:
        smoking_alcohol = st.selectbox("Smoking/Alcohol", ["no", "yes"])
        gender = st.selectbox("Gender", ["male", "female", "other"])

    st.markdown("---")

    if st.button("Predict Health Risk", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "sleep_duration": sleep_duration,
            "heart_rate": heart_rate,
            "bmi": bmi,
            "calorie_expenditure": calorie_expenditure,
            "step_count": step_count,
            "exercise_duration": exercise_duration,
            "water_intake": water_intake,
            "diet_type": diet_type,
            "stress_level": stress_level,
            "sleep_quality": sleep_quality,
            "physical_activity_level": physical_activity_level,
            "smoking_alcohol": smoking_alcohol,
            "gender": gender,
        }])

        try:
            labels, probs, classes = predict(input_df, model_data)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        label = labels[0]
        prob_dict = {c: float(p) for c, p in zip(classes, probs[0])}

        st.markdown("### Prediction Result")

        risk_class = f"risk-{label}"
        if label == "at-risk":
            emoji = "⚠️"
            desc = "This student shows signs of being at-risk and should seek medical attention."
        elif label == "unhealthy":
            emoji = "🟠"
            desc = "This student's health indicators suggest an unhealthy lifestyle."
        else:
            emoji = "✅"
            desc = "This student appears to be in good health!"

        st.markdown(f"""
        <div class="metric-card {risk_class}">
            <h3>{emoji} {label.upper()}</h3>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### Class Probabilities")

        prob_df = pd.DataFrame({
            "Class": list(prob_dict.keys()),
            "Probability": list(prob_dict.values())
        }).sort_values("Probability", ascending=False)

        for _, row in prob_df.iterrows():
            st.markdown(f"**{row['Class']}**")
            st.progress(float(row["Probability"]))
            st.caption(f"{row['Probability']:.1%}")


def batch_prediction(model_data):
    st.subheader("Upload CSV for Batch Prediction")
    st.markdown(
        "Upload a CSV file with columns: `sleep_duration`, `heart_rate`, `bmi`, "
        "`calorie_expenditure`, `step_count`, `exercise_duration`, `water_intake`, "
        "`diet_type`, `stress_level`, `sleep_quality`, `physical_activity_level`, "
        "`smoking_alcohol`, `gender`"
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.markdown(f"**Uploaded:** {len(df)} rows, {len(df.columns)} columns")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Predicting..."):
                try:
                    labels, probs, classes = predict(df, model_data)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    return

            result_df = df.copy()
            result_df["prediction"] = labels
            for i, c in enumerate(classes):
                result_df[f"prob_{c}"] = probs[:, i]

            st.markdown("### Results")
            st.dataframe(result_df.head(50), use_container_width=True)

            st.markdown("#### Prediction Distribution")
            dist = pd.Series(labels).value_counts()
            st.bar_chart(dist)

            csv = result_df.to_csv(index=False)
            st.download_button(
                "Download Predictions",
                csv,
                "predictions.csv",
                "text/csv",
                use_container_width=True,
            )


def main():
    st.markdown(
        '<div class="main-header">🏥 Student Health Risk Predictor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Predict whether a student is at-risk, unhealthy, or fit</div>',
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error(
            f"Models not found at `{MODEL_PATH}`. "
            "Run `python inference.py` first to train and save models."
        )
        return

    try:
        model_data = load_model()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    with tab1:
        single_prediction(model_data)

    with tab2:
        batch_prediction(model_data)


if __name__ == "__main__":
    main()
