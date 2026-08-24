import pandas as pd

# Load dataset
df = pd.read_csv("../data/processed/final_dataset.csv")

# Correlation with Production
correlation = df[
    ["Production", "Area", "Yield"]
].corr()

print("Correlation Matrix:")
print(correlation)

print("\nCorrelation with Production:")
print(
    correlation["Production"]
    .sort_values(ascending=False)
)