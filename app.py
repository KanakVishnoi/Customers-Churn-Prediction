
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# -------------------- PAGE CONFIGURATION --------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# -------------------- LOAD SAVED MODEL --------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.joblib"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a telecom customer is likely to leave "
    "the service using a trained machine-learning model."
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model files are missing. Please check the model folder.")
    st.stop()

try:
    model, metadata = load_model()
    threshold = metadata["decision_threshold"]
except Exception as error:
    st.error(f"Unable to load the saved model: {error}")
    st.stop()

st.caption(f"Decision threshold: {threshold:.2%}")

# -------------------- CUSTOMER INPUTS --------------------
st.subheader("Enter Customer Details")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0,
            max_value=2000.0, value=70.0, step=1.0,
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0,
            max_value=100000.0, value=840.0, step=10.0,
        )

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn", use_container_width=True
    )

# -------------------- PREDICTION --------------------
if submitted:
    customer = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )

    try:
        churn_probability = float(model.predict_proba(customer)[0][1])
        prediction = int(churn_probability >= threshold)

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Predicted Churn Probability",
                f"{churn_probability:.1%}",
            )

        with result_col2:
            if prediction == 1:
                st.error("⚠️ Prediction: Customer Likely to Churn")
            else:
                st.success("✅ Prediction: Customer Likely to Stay")

        st.progress(churn_probability)
        st.caption(
            "This is a machine-learning estimate, not a guarantee "
            "of future customer behavior."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
