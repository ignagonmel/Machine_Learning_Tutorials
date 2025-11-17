import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import os


# 1. Load dataset

input_csv = "dataset_additional.csv"

df = pd.read_csv(input_csv)
print(f"Loaded dataset '{input_csv}' with {len(df)} samples and {df.shape[1]} columns.")

# Separate features (X) and target (y)
X = df.drop(columns=["action"])
y = df["action"]


# 2. Mutual Information (MI) feature selection

print("\n Performing Mutual Information (MI) feature selection...")

# Compute MI between each feature and the target (action)
mi_scores = mutual_info_classif(X, y, random_state=42)

# Create DataFrame with scores
mi_df = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores})
mi_df = mi_df.sort_values(by="MI_Score", ascending=False)
print("\n Mutual Information Scores:")
print(mi_df)

# Select features above median importance
mi_threshold = np.median(mi_scores)
selected_mi = X.columns[mi_scores > mi_threshold]
print(f"\n Selected {len(selected_mi)} features using MI: {list(selected_mi)}")

# Save MI dataset
df_mi = df[["action"] + list(selected_mi)]
os.makedirs("DATA_T3", exist_ok=True)
df_mi.to_csv("DATA_T3/dataset_mi.csv", index=False)
print(" Saved 'datasets/dataset_mi.csv'")


# 3. Recursive Feature Elimination (RFE)

print("\n Performing Recursive Feature Elimination (RFE)...")

# Use a Random Forest as estimator (robust for feature selection)
model = RandomForestClassifier(n_estimators=50, random_state=42)
selector = RFE(model, n_features_to_select=max(3, X.shape[1] // 2))
selector.fit(X, y)

# Get selected features
selected_rfe = X.columns[selector.support_]
print(f"\n Selected {len(selected_rfe)} features using RFE: {list(selected_rfe)}")

# Save RFE dataset
df_rfe = df[["action"] + list(selected_rfe)]
df_rfe.to_csv("DATA_T3/dataset_rfe.csv", index=False)
print(" Saved 'datasets/dataset_rfe.csv'")

print("\n Feature selection completed successfully!")
