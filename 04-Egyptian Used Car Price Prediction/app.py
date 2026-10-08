import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================================
# Load Model and Preprocessing Objects
# ==========================================

model = joblib.load("car_price_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")


# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("hatla2ee_scraped_data.csv")


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Egyptian Used Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)


# ==========================================
# Title
# ==========================================

st.title("🚗 Egyptian Used Car Price Prediction")

st.write(
    "Enter the car information to predict its expected price in EGP."
)


# ==========================================
# Get Categories from Dataset
# ==========================================

makes = sorted(
    df["Make"].dropna().unique().tolist()
)

models = sorted(
    df["Model"].dropna().unique().tolist()
)

colors = sorted(
    df["Color"].dropna().unique().tolist()
)

cities = sorted(
    df["City"].dropna().unique().tolist()
)


# ==========================================
# User Inputs
# ==========================================

st.subheader("Car Information")

col1, col2 = st.columns(2)


with col1:

    make = st.selectbox(
        "Make",
        makes
    )

    model_name = st.selectbox(
        "Model",
        models
    )

    color = st.selectbox(
        "Color",
        colors
    )

    city = st.selectbox(
        "City",
        cities
    )


with col2:

    year = st.number_input(
        "Year",
        min_value=1990,
        max_value=2026,
        value=2020,
        step=1
    )

    mileage = st.number_input(
        "Mileage (Km)",
        min_value=0,
        value=50000,
        step=1000
    )


automatic = st.selectbox(
    "Automatic Transmission",
    ["Yes", "No"]
)

air_conditioner = st.selectbox(
    "Air Conditioner",
    ["Yes", "No"]
)

power_steering = st.selectbox(
    "Power Steering",
    ["Yes", "No"]
)

remote_control = st.selectbox(
    "Remote Control",
    ["Yes", "No"]
)


# ==========================================
# Prediction
# ==========================================

if st.button("Predict Price 🚀"):

    input_data = pd.DataFrame({

        "Mileage": [mileage],

        "Year": [year],

        "Color": [color],

        "Make": [make],

        "Model": [model_name],

        "City": [city],

        "Automatic Transmission": [automatic],

        "Air Conditioner": [air_conditioner],

        "Power Steering": [power_steering],

        "Remote Control": [remote_control]
    })


    # ======================================
    # Feature Lists
    # ======================================

    numerical_features = [
        "Mileage",
        "Year"
    ]

    categorical_features = [
        "Color",
        "Make",
        "Model",
        "City",
        "Automatic Transmission",
        "Air Conditioner",
        "Power Steering",
        "Remote Control"
    ]


    # ======================================
    # Numerical Preprocessing
    # ======================================

    input_numeric_scaled = scaler.transform(
        input_data[numerical_features]
    )


    # ======================================
    # Categorical Preprocessing
    # ======================================

    input_encoded = encoder.transform(
        input_data[categorical_features]
    )


    # ======================================
    # Combine Features
    # ======================================

    input_processed = np.hstack([
        input_numeric_scaled,
        input_encoded
    ])


    # ======================================
    # Prediction
    # ======================================

    prediction = model.predict(
        input_processed
    )[0]


    # ======================================
    # Display Result
    # ======================================

    st.success(
        f"💰 Estimated Price: {prediction:,.0f} EGP"
    )
    