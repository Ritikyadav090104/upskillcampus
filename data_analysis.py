import pandas as pd

# Load all datasets
produce = pd.read_csv("../data/raw/produce.csv")
cost_yield = pd.read_csv("../data/raw/datafile (1).csv")
production_data = pd.read_csv("../data/raw/datafile (2).csv")
crop_variety = pd.read_csv("../data/raw/datafile (3).csv")
crop_index = pd.read_csv("../data/raw/datafile.csv")


# --------------------------------------------------
# 1. PRODUCE DATASET
# --------------------------------------------------

print("\n========== PRODUCE DATASET ==========")

print("Shape:", produce.shape)
print("Columns:")
print(produce.columns.tolist())

print("\nFirst 5 rows:")
print(produce.head())


# --------------------------------------------------
# 2. COST & YIELD DATASET
# --------------------------------------------------

print("\n========== COST & YIELD DATASET ==========")

print("Shape:", cost_yield.shape)
print("Columns:")
print(cost_yield.columns.tolist())

print("\nFirst 5 rows:")
print(cost_yield.head())


# --------------------------------------------------
# 3. PRODUCTION DATASET
# --------------------------------------------------

print("\n========== PRODUCTION DATASET ==========")

print("Shape:", production_data.shape)
print("Columns:")
print(production_data.columns.tolist())

print("\nFirst 5 rows:")
print(production_data.head())


# --------------------------------------------------
# 4. CROP VARIETY DATASET
# --------------------------------------------------

print("\n========== CROP VARIETY DATASET ==========")

print("Shape:", crop_variety.shape)
print("Columns:")
print(crop_variety.columns.tolist())

print("\nFirst 5 rows:")
print(crop_variety.head())


# --------------------------------------------------
# 5. CROP INDEX DATASET
# --------------------------------------------------

print("\n========== CROP INDEX DATASET ==========")

print("Shape:", crop_index.shape)
print("Columns:")
print(crop_index.columns.tolist())

print("\nFirst 5 rows:")
print(crop_index.head())

# ==========================================================
# CREATE ML-READY DATASET
# ==========================================================

import re

df = production_data.copy()

# Clean column names
df.columns = df.columns.str.strip()

# Clean crop names
df["Crop"] = df["Crop"].astype(str).str.strip()

# Convert wide format → long format
records = []

years = ["2006-07", "2007-08", "2008-09", "2009-10", "2010-11"]

for _, row in df.iterrows():

    crop = row["Crop"]

    for year in years:

        records.append({
            "Crop": crop,
            "Year": year,
            "Production": row[f"Production {year}"],
            "Area": row[f"Area {year}"],
            "Yield": row[f"Yield {year}"]
        })

final_df = pd.DataFrame(records)

# Convert numeric columns
numeric_columns = ["Production", "Area", "Yield"]

for col in numeric_columns:
    final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

# Remove missing values
final_df = final_df.dropna()

# Remove duplicate rows
final_df = final_df.drop_duplicates()

# Convert year into numerical value
final_df["Year"] = final_df["Year"].str[:4].astype(int)

# Save processed dataset
final_df.to_csv(
    "data/processed/final_dataset.csv",
    index=False
)

print("\n====================================")
print("FINAL ML DATASET")
print("====================================")

print("Shape:", final_df.shape)

print("\nColumns:")
print(final_df.columns.tolist())

print("\nFirst 10 rows:")
print(final_df.head(10))

print("\nMissing values:")
print(final_df.isnull().sum())

print("\nDataset saved successfully!")