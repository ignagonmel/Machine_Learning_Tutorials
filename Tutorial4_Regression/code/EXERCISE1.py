import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error

# LOAD DATA
train_data = pd.read_csv("Datasets/training_data.csv")
test_data = pd.read_csv("Datasets/test_data.csv")

X_train = train_data.drop(columns=["Action"])
y_train = train_data["Action"]
X_test = test_data.drop(columns=["Action"])
y_test = test_data["Action"]

# DEFINE PARAMETER GRID
param_grid = {
    "criterion": ["squared_error", "absolute_error"],
    "splitter": ["best", "random"],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5],
    "max_features": [None, "sqrt"]
}

param_combinations = list(product(*param_grid.values()))
print(f"Total models to evaluate: {len(param_combinations)}")

# CROSS VALIDATION TRAINING
results = []

for i, combo in enumerate(param_combinations, start=1):
    params = dict(zip(param_grid.keys(), combo))
    model = DecisionTreeRegressor(random_state=42, **params)

    # Perform 5-fold cross-validation
    mse_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    mae_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    rmse_scores = np.sqrt(mse_scores)

    results.append({
        "Model": f"Model_{i}",
        **params,
        "MSE": np.mean(mse_scores),
        "RMSE": np.mean(rmse_scores),
        "MAE": np.mean(mae_scores)
    })

results_df = pd.DataFrame(results).sort_values(by="MSE", ascending=True).reset_index(drop=True)
results_df.to_csv("results_ex1_1.csv", index=False)
print("\nEXERCISE 1.1 COMPLETED — Results saved to 'results_ex1_1.csv'")

top3 = results_df.head(3)
print("\n===== TOP 3 MODELS (LOWEST MSE) =====")
print(top3.to_string(index=False))

# TEST PERFORMANCE OF TOP 3 MODELS
test_results = []

for _, row in top3.iterrows():
    params = {
        "criterion": row["criterion"],
        "splitter": row["splitter"],
        "max_depth": int(row["max_depth"]),
        "min_samples_split": int(row["min_samples_split"]),
        "max_features": None if row["max_features"] == "None" else row["max_features"]
    }

    model = DecisionTreeRegressor(random_state=42, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    test_results.append({
        "Model": row["Model"],
        **params,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    })

test_df = pd.DataFrame(test_results).sort_values(by="MSE", ascending=True).reset_index(drop=True)
print("\nEXERCISE 1.2 — TEST RESULTS")
print(test_df.to_string(index=False))

# RETRAIN BEST MODEL
best_params = test_df.iloc[0].to_dict()
best_model = DecisionTreeRegressor(
    random_state=42,
    criterion=best_params["criterion"],
    splitter=best_params["splitter"],
    max_depth=int(best_params["max_depth"]),
    min_samples_split=int(best_params["min_samples_split"]),
    max_features=None if best_params["max_features"] == "None" else best_params["max_features"]
)

best_model.fit(X_train, y_train)
y_pred_final = best_model.predict(X_test)

final_mse = mean_squared_error(y_test, y_pred_final)
final_rmse = np.sqrt(final_mse)
final_mae = mean_absolute_error(y_test, y_pred_final)

print("\n===== EXERCISE 1.3: FINAL MODEL (NO CROSS-VALIDATION) =====")
print(f"Criterion: {best_params['criterion']}, Splitter: {best_params['splitter']}, "
      f"Max Depth: {best_params['max_depth']}, Min Samples Split: {best_params['min_samples_split']}, "
      f"Max Features: {best_params['max_features']}")
print(f"MSE = {final_mse:.4f} | RMSE = {final_rmse:.4f} | MAE = {final_mae:.4f}")
