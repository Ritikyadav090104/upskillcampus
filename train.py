import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("../data/processed/final_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# --------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop("Production", axis=1)

y = df["Production"]


# --------------------------------------------------
# 3. TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 4. ENCODE CROP COLUMN
# --------------------------------------------------

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


# --------------------------------------------------
# 5. GET NUMERICAL FEATURES
# --------------------------------------------------

numeric_train = X_train[
    ["Year", "Area", "Yield"]
].values

numeric_test = X_test[
    ["Year", "Area", "Yield"]
].values


# --------------------------------------------------
# 6. COMBINE CATEGORICAL + NUMERICAL FEATURES
# --------------------------------------------------

import numpy as np

X_train_final = np.hstack(
    [crop_train, numeric_train]
)

X_test_final = np.hstack(
    [crop_test, numeric_test]
)


# --------------------------------------------------
# 7. CREATE MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# --------------------------------------------------
# 8. TRAIN MODEL
# --------------------------------------------------

model.fit(
    X_train_final,
    y_train
)


# --------------------------------------------------
# 9. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test_final)


# --------------------------------------------------
# 10. EVALUATE MODEL
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# --------------------------------------------------
# 11. SAVE MODEL
# --------------------------------------------------

joblib.dump(
    model,
    "../models/best_model.pkl"
)

joblib.dump(
    encoder,
    "../models/encoder.pkl"
)

print("\nModel saved successfully!")