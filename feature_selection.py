import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# Load processed dataset
df = pd.read_csv("outputs/processed/processed_dataset.csv")

selector = VarianceThreshold(threshold=0)

X = selector.fit_transform(df)

selected_columns = df.columns[selector.get_support()]

print("Selected Features:\n")

for feature in selected_columns:
    print(feature)

pd.DataFrame({
    "Selected Features": selected_columns
}).to_csv(
    "outputs/reports/selected_features.csv",
    index=False
)

print("\nFeature selection completed.")

import pandas as pd
import matplotlib.pyplot as plt

# Load processed dataset
df = pd.read_csv("outputs/processed/processed_dataset.csv")

# Correlation matrix
corr = df.corr(numeric_only=True)

# Save CSV
corr.to_csv("outputs/reports/correlation_matrix.csv")

# Plot
plt.figure(figsize=(14, 10))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/plots/feature_correlation.png", dpi=300)
plt.close()

print("✓ Correlation matrix saved.")