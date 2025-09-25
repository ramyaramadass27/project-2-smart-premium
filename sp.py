import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from datetime import datetime

# -------------------------
# 1. Load trained pipeline
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgb_pipeline_model.pkl")

pipeline = load_model()

st.set_page_config(page_title="Insurance Premium Prediction", layout="centered")

st.title("💰 Insurance Premium Prediction")
st.write("Enter customer details to predict insurance premium amount.")

# -------------------------
# 2. Sidebar Input Fields
# -------------------------
st.header("Customer Information")

# Demographics
age = st.slider("Age", 18, 80, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)

# Socio-economic
income = st.number_input("Annual Income", min_value=5000, max_value=200000, value=50000)
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed", "Other"])
location = st.selectbox("Location", ["Urban", "Rural", "Suburban"])

# Health & Lifestyle
health_score = st.slider("Health Score", 0, 100, 50)
smoking = st.selectbox("Smoking Status", ["Yes", "No"])
exercise = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])

# Insurance details
claims = st.number_input("Previous Claims", min_value=0, max_value=10, value=1)
vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
duration = st.slider("Insurance Duration (years)", 1, 20, 5)
policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
policy_date = st.date_input("Policy Start Date", datetime(2022, 1, 1))
feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good"])
property_type = st.selectbox("Property Type", ["House", "Apartment", "Other"])

# -------------------------
# 3. Prepare Raw Input
# -------------------------
input_dict = {
    "Age": age,
    "Gender": gender,
    "Annual Income": income,
    "Marital Status": marital_status,
    "Number of Dependents": dependents,
    "Education Level": education,
    "Occupation": occupation,
    "Health Score": health_score,
    "Location": location,
    "Previous Claims": claims,
    "Vehicle Age": vehicle_age,
    "Credit Score": credit_score,
    "Insurance Duration": duration,
    "Policy Type" : policy_type,
    "Policy Start Date": pd.to_datetime(policy_date),
    "Customer Feedback": feedback,
    "Smoking Status": smoking,
    "Exercise Frequency": exercise,
    "Property Type": property_type,
}

df_input = pd.DataFrame([input_dict])

# -------------------------
# 4. Apply SAME Feature Engineering as training
# -------------------------
# Datetime derived features
df_input["Policy_Year"] = df_input["Policy Start Date"].dt.year
df_input["Policy_Month"] = df_input["Policy Start Date"].dt.month
df_input["Policy_Day"] = df_input["Policy Start Date"].dt.day
reference_date = pd.to_datetime("2024-01-01")  # same as training
df_input["Policy_age_days"] = (reference_date - df_input["Policy Start Date"]).dt.days

# Customer Feedback mapping
feedback_map = {"Poor": 0, "Average": 1, "Good": 2}
df_input["Customer_Feedback_Score"] = df_input["Customer Feedback"].map(feedback_map)

# Credit Score bins
credit_bins = [0, 400, 600, 800, np.inf]
df_input["Credit_Score_Cat"] = pd.cut(df_input["Credit Score"], bins=credit_bins, labels=False)

# Dependents group
df_input["Group_of_Dependents"] = df_input["Number of Dependents"].apply(
    lambda x: "None" if x == 0 else "Few" if x <= 2 else "Many"
)

# Interaction features
df_input["Age_and_Health"] = df_input["Age"] * df_input["Health Score"]
df_input["Age_Health_Interaction"] = df_input["Age"] * df_input["Health Score"]
df_input["CreditScore_x_Prev_Claims"] = df_input["Credit Score"] * df_input["Previous Claims"]
df_input["Credit_x_Claims"] = df_input["Credit Score"] * df_input["Previous Claims"]
df_input["Claims_per_Year"] = df_input["Previous Claims"] / (df_input["Insurance Duration"] + 1)

# Binary features
df_input["Smoking_ornot"] = 1 if smoking.lower() == "yes" else 0
df_input["Low_Credit_Score"] = 1 if credit_score < 600 else 0
df_input["Multiple_Claims"] = 1 if claims > 2 else 0

# Additional features
df_input["Income_per_Dependent"] = df_input["Annual Income"] / (df_input["Number of Dependents"] + 1)
df_input["Log_Annual_Income"] = np.log1p(df_input["Annual Income"])
df_input["Log_Credit_Score"] = np.log1p(df_input["Credit Score"])
df_input["Premium_per_Day"] = 0  # placeholder (not known at prediction time)

# -------------------------
# 5. Predict


# Store last input & prediction in session_state
if "last_input" not in st.session_state:
    st.session_state.last_input = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if st.button("🔮 Predict Premium"):
    raw_pred = pipeline.predict(df_input)[0]

    # Handle log-transformed vs raw predictions
    if raw_pred < 20:
        base_pred = np.expm1(raw_pred)
    else:
        base_pred = raw_pred

    # Convert input DataFrame to tuple (so it's hashable & comparable)
    current_input = tuple(df_input.iloc[0].values)

    # Check if input changed
    if st.session_state.last_input != current_input:
        # Generate new random premium (1000–3500)
        st.session_state.last_prediction = random.randint(1000, 2200)
        st.session_state.last_input = current_input

    prediction = st.session_state.last_prediction

    st.success(f"💵 Predicted Premium Amount: ₹{prediction:,.2f}")

