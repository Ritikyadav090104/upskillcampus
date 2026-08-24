import pandas as pd
import numpy as np
import joblib


# Load model and encoder
model = joblib.load("../models/best_model.pkl")
encoder = joblib.load("../models/encoder.pkl")


# User input
crop = input("Enter crop name: ")
year = int(input("Enter year: "))
area = float(input("Enter area: "))


# Create input dataframe
input_data = pd.DataFrame({
    "Crop": [crop],
    "Year": [year],
    "Area": [area]
})


# Encode crop
crop_encoded = encoder.transform(
    input_data[["Crop"]]
)


# Numerical features
numeric_data = input_data[
    ["Year", "Area"]
].values


# Combine features
final_input = np.hstack(
    [crop_encoded, numeric_data]
)


# Predict
prediction = model.predict(
    final_input
)


print("\n==============================")
print("PREDICTION")
print("==============================")

print(
    "Predicted Production:",
    round(prediction[0], 2)
)