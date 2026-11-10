import pandas as pd
import numpy as np

DATASET_PATH = (
    "/Users/kratikay/Desktop/College/"
    "MINOR PROJECT/data/merged_dataset.csv"
)

df = pd.read_csv(DATASET_PATH)

print("\n========== ROAD NETWORK DIAGNOSTICS ==========")

# 1. Check required columns
required_columns = [
    "Road_ID",
    "u",
    "v",
    "oneway",
    "length",
    "Latitude",
    "Longitude",
    "Crime_Index",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\n1. Required columns are available.")

# 2. Check node identifiers
print("\n2. NODE IDENTIFIERS")
print("Missing u values:", df["u"].isna().sum())
print("Missing v values:", df["v"].isna().sum())

# 3. Check road lengths
print("\n3. ROAD LENGTHS")
print("Minimum length:", df["length"].min())
print("Maximum length:", df["length"].max())
print("Average length:", round(df["length"].mean(), 2))

# 4. Inspect coordinate ranges
print("\n4. ROAD COORDINATES")
print("Latitude range:",
      df["Latitude"].min(),
      "to",
      df["Latitude"].max())

print("Longitude range:",
      df["Longitude"].min(),
      "to",
      df["Longitude"].max())

# 5. Check repeated node pairs
print("\n5. REPEATED NODE PAIRS")
print(
    "Repeated (u, v) records:",
    df.duplicated(subset=["u", "v"]).sum()
)

# 6. Estimate whether recorded lengths are consistent
# with straight-line distance between road-record coordinates.
#
# IMPORTANT:
# Latitude/Longitude appears to represent one location
# per road record, not necessarily either endpoint.
# Therefore, this is only a diagnostic, not a validation.

R = 6371000

lat1 = np.radians(df["Latitude"].to_numpy()[:-1])
lat2 = np.radians(df["Latitude"].to_numpy()[1:])
lon1 = np.radians(df["Longitude"].to_numpy()[:-1])
lon2 = np.radians(df["Longitude"].to_numpy()[1:])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat1)
    * np.cos(lat2)
    * np.sin(dlon / 2) ** 2
)

a = np.clip(a, 0, 1)

distances = 2 * R * np.arcsin(np.sqrt(a))

print("\n6. COORDINATE SPACING DIAGNOSTIC")
print(
    "Median distance between consecutive CSV rows:",
    round(float(np.median(distances)), 2),
    "metres"
)

print("\n7. SAMPLE ROAD RECORDS")
print(
    df[
        [
            "Road_ID",
            "u",
            "v",
            "length",
            "Latitude",
            "Longitude",
            "oneway",
            "Crime_Index",
        ]
    ].head(10).to_string(index=False)
)

print("\n========== DIAGNOSTICS COMPLETE ==========")