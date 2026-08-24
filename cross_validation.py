import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


# Load data
df = pd.read_csv("../data/processed/final_dataset.csv")


# Features and target
X = df.drop("Production", axis=1)
y = df["Production"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


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


# Numerical columns
numeric_train = X_train[
    ["Year", "Area", "Yield"]
].values

numeric_test = X_test[
    ["Year", "Area", "Yield"]
].values


# Final data
X_train_final = np.hstack(
    [crop_train, numeric_train]
)

X_test_final = np.hstack(
    [crop_test, numeric_test]
)


# Create model
model = LinearRegression()


# Cross validation
scores = cross_val_score(
    model,
    X_train_final,
    y_train,
    cv=5,
    scoring="r2"
)


print("Cross Validation R2 Scores:")
print(scores)

print("\nAverage CV R2 Score:")
print(scores.mean())