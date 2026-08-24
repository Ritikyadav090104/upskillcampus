import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load data
df = pd.read_csv("../data/processed/final_dataset.csv")


# Features and target
X = df[["Crop", "Year", "Area"]]
y = df["Production"]


# Train = 2006-2009
# Test = 2010
train_data = df[df["Year"] < 2010]
test_data = df[df["Year"] == 2010]


X_train = train_data[["Crop", "Year", "Area"]]
y_train = train_data["Production"]

X_test = test_data[["Crop", "Year", "Area"]]
y_test = test_data["Production"]


# Encode Crop
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

crop_train = encoder.fit_transform(
    X_train[["Crop"]]
)

crop_test = encoder.transform(
    X_test[["Crop"]]
)


# Numerical features
numeric_train = X_train[
    ["Year", "Area"]
].values

numeric_test = X_test[
    ["Year", "Area"]
].values


# Final training data
X_train_final = np.hstack(
    [crop_train, numeric_train]
)

X_test_final = np.hstack(
    [crop_test, numeric_test]
)


# Create model
model = LinearRegression()


# Train
model.fit(
    X_train_final,
    y_train
)


# Prediction
y_pred = model.predict(X_test_final)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==============================")
print("FINAL MODEL RESULTS")
print("==============================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))


# Save model
joblib.dump(
    model,
    "../models/best_model.pkl"
)

joblib.dump(
    encoder,
    "../models/encoder.pkl"
)

print("\nModel saved successfully!")