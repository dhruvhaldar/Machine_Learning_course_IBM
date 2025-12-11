import pandas as pd
import numpy as np
from sklearn import linear_model
import joblib
import os

# Set working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the dataset
try:
    df = pd.read_csv("FuelConsumption.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: FuelConsumption.csv not found.")
    exit(1)

# Select features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# Define training data
# We are doing Simple Linear Regression: Engine Size vs CO2 Emissions
train_x = np.asanyarray(cdf[['ENGINESIZE']])
train_y = np.asanyarray(cdf[['CO2EMISSIONS']])

# Train the model
print("Training the model...")
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

# Print coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# Save the model
model_filename = 'simple_linear_model.pkl'
joblib.dump(regr, model_filename)
print(f"Model saved to {model_filename}")
