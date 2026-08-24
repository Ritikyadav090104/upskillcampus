import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("../data/processed/final_dataset.csv")

print("Dataset Shape:", df.shape)


# --------------------------------------------------
# 2. FEATURES AND TARGET
# --------------------------------------------------

X = df[["Crop", "Year", "Area"]]
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
# 4. ENCODE CROP
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
# 5. NUMERICAL FEATURES
# --------------------------------------------------

numeric_train = X_train[
    ["Year", "Area"]
].values

numeric_test = X_test[
    ["Year", "Area"]
].values
# --------------------------------------------------
# 6. FINAL TRAIN AND TEST DATA
# --------------------------------------------------

X_train_final = np.hstack(
    [crop_train, numeric_train]
)

X_test_final = np.hstack(
    [crop_test, numeric_test]
)


# --------------------------------------------------
# 7. CREATE MODELS
# --------------------------------------------------

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
}


# --------------------------------------------------
# 8. TRAIN AND COMPARE MODELS
# --------------------------------------------------

for name, model in models.items():

    model.fit(
        X_train_final,
        y_train
    )

    y_pred = model.predict(
        X_test_final
    )

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
    print(name)
    print("==============================")

    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))