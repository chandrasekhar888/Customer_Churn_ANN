import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

from tensorflow.keras.models import load_model


# =========================================================
# 1. FIND THE FOLDER WHERE app.py IS LOCATED
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# 2. LOAD TRAINED MODEL
# =========================================================

model_path = os.path.join(BASE_DIR, "churn_model.h5")

model = load_model(model_path)


# =========================================================
# 3. LOAD PREPROCESSING OBJECTS
# =========================================================

with open(
    os.path.join(BASE_DIR, "label_encoder_gender.pkl"),
    "rb"
) as file:
    label_encoder_gender = pickle.load(file)


with open(
    os.path.join(BASE_DIR, "onehot_encoder_geo.pkl"),
    "rb"
) as file:
    onehot_encoder_geo = pickle.load(file)


with open(
    os.path.join(BASE_DIR, "scaler.pkl"),
    "rb"
) as file:
    scaler = pickle.load(file)


# =========================================================
# 4. STREAMLIT PAGE
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


st.title("📊 Customer Churn Prediction")

st.write(
    "Enter the customer's details below to predict "
    "whether the customer is likely to leave the bank."
)


# =========================================================
# 5. USER INPUTS
# =========================================================

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)


geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)


tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)


balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=75000.0
)


num_of_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2
)


has_cr_card = st.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)


is_active_member = st.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)


estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=90000.0
)


# =========================================================
# 6. PREDICTION BUTTON
# =========================================================

if st.button("Predict Churn"):

    # -----------------------------------------------------
    # Convert Yes/No values to 1/0
    # -----------------------------------------------------

    has_cr_card_value = 1 if has_cr_card == "Yes" else 0

    is_active_member_value = (
        1 if is_active_member == "Yes" else 0
    )


    # -----------------------------------------------------
    # Gender Label Encoding
    # -----------------------------------------------------

    gender_encoded = label_encoder_gender.transform(
        [gender]
    )[0]


    # -----------------------------------------------------
    # Geography One-Hot Encoding
    # -----------------------------------------------------

    geography_encoded = onehot_encoder_geo.transform(
        [[geography]]
    ).toarray()


    # -----------------------------------------------------
    # Create DataFrame for numerical/categorical features
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        {
            "CreditScore": [credit_score],
            "Gender": [gender_encoded],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card_value],
            "IsActiveMember": [is_active_member_value],
            "EstimatedSalary": [estimated_salary],
        }
    )


    # -----------------------------------------------------
    # Add Geography encoded columns
    # -----------------------------------------------------

    geography_columns = (
        onehot_encoder_geo.get_feature_names_out(
            ["Geography"]
        )
    )


    geography_df = pd.DataFrame(
        geography_encoded,
        columns=geography_columns
    )


    # -----------------------------------------------------
    # Combine everything
    # -----------------------------------------------------

    final_data = pd.concat(
        [input_data, geography_df],
        axis=1
    )


    # -----------------------------------------------------
    # Scale the data
    # -----------------------------------------------------

    final_data_scaled = scaler.transform(final_data)


    # -----------------------------------------------------
    # ANN Prediction
    # -----------------------------------------------------

    prediction = model.predict(
        final_data_scaled,
        verbose=0
    )


    probability = prediction[0][0]


    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    st.subheader("Prediction Result")

    st.write(
        f"Churn Probability: **{probability:.2%}**"
    )


    if probability >= 0.5:

        st.error(
            "⚠️ Customer is likely to churn."
        )

    else:

        st.success(
            "✅ Customer is likely to stay."
        )