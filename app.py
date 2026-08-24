import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoder
model = joblib.load("models/best_model.pkl")
encoder = joblib.load("models/encoder.pkl")

# Page settings & Modern Layout
st.set_page_config(
    page_title="Agriculture Production Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/final_dataset.csv")

df = load_data()

# Sidebar for Navigation & Info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/agriculture.png", width=80)
    st.title("AgriPredict")
    st.markdown("---")
    st.info("This application uses a Machine Learning model to forecast agricultural crop production based on historical data, area size, and temporal factors.")
    st.markdown("### Developed for Smart Agriculture")

# Main Header Section
st.title("🌾 Agriculture Crop Production Predictor")
st.markdown("Get accurate data-driven forecasts for crop production to optimize agricultural planning.")
st.markdown("---")

# Layout using columns for input and quick metrics
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("📋 Enter Crop Parameters")
    
    crop_list = sorted(df["Crop"].unique())
    
    crop = st.selectbox(
        "Select Crop",
        crop_list,
        help="Choose the specific crop you want to predict production for."
    )

    col_yr, col_ar = st.columns(2)
    with col_yr:
        year = st.number_input(
            "Enter Year",
            min_value=2006,
            max_value=2030,
            value=2010,
            help="Target year for prediction."
        )

    with col_ar:
        area = st.number_input(
            "Enter Area (Hectares)",
            min_value=0.0,
            value=100.0,
            step=10.0,
            help="Total farming area allocated."
        )

    st.markdown("")
    predict_clicked = st.button("🚀 Predict Production")

with col2:
    st.subheader("📊 Dataset Quick Stats")
    st.markdown(f"""
    <div class="metric-card">
        <h4 style='color: #2e7d32; margin-bottom: 0;'>Total Crops</h4>
        <h2>{len(crop_list)}</h2>
        <hr style='margin: 10px 0;'>
        <h4 style='color: #2e7d32; margin-bottom: 0;'>Year Range</h4>
        <p style='font-size: 1.2rem; font-weight: bold; margin-top: 5px;'>2006 - 2030</p>
    </div>
    """, unsafe_allow_html=True)

# Prediction Result Section
if predict_clicked:
    with st.spinner("Calculating production forecast..."):
        # Create input dataframe
        input_data = pd.DataFrame({
            "Crop": [crop],
            "Year": [year],
            "Area": [area]
        })

        # Encode crop
        crop_encoded = encoder.transform(input_data[["Crop"]])

        # Numerical features
        numeric_data = input_data[["Year", "Area"]].values

        # Combine features
        final_input = np.hstack([crop_encoded, numeric_data])

        # Prediction
        prediction = model.predict(final_input)

    st.markdown("---")
    st.subheader("📈 Prediction Result")
    
    # Display styled success card
    st.success(f"### Predicted Production for **{crop}** in **{year}**: `{prediction[0]:,.2f}` Units")