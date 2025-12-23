import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Determine the directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths
model_path = os.path.join(BASE_DIR, 'simple_linear_model.pkl')
data_path = os.path.join(BASE_DIR, 'FuelConsumption.csv')

# Load the model
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Please run train_model.py first.")
    st.stop()

# Load dataset for visualization
try:
    df = pd.read_csv(data_path)
    cdf = df[['ENGINESIZE', 'CO2EMISSIONS']]
except FileNotFoundError:
    st.warning(f"FuelConsumption.csv not found at {data_path}. Visualization will be limited.")
    cdf = None

st.title("CO2 Emissions Predictor")
st.write("This app predicts CO2 emissions based on the engine size of a vehicle using a Simple Linear Regression model.")

# User input
st.subheader("Input Parameters")
engine_size = st.number_input("Engine Size (L)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

if st.button("Predict"):
    # Predict
    input_data = np.array([[engine_size]])
    prediction = model.predict(input_data)
    predicted_emission = prediction[0][0]

    st.subheader("Prediction")
    st.success(f"Predicted CO2 Emissions: {predicted_emission:.2f} g/km")

    # Visualization
    if cdf is not None:
        st.subheader("Visualization")
        fig, ax = plt.subplots()

        # Plot training data
        ax.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue', label='Training Data', alpha=0.5)

        # Plot regression line
        train_x = np.asanyarray(cdf[['ENGINESIZE']])
        ax.plot(train_x, model.coef_[0][0]*train_x + model.intercept_[0], '-r', label='Regression Line')

        # Plot user input
        ax.scatter(engine_size, predicted_emission, color='green', s=100, label='Your Input', zorder=5)

        ax.set_xlabel("Engine size")
        ax.set_ylabel("Emission")
        ax.legend()

        st.pyplot(fig)
