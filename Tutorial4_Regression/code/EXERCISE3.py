import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import matplotlib.pyplot as plt

train_data = pd.read_csv("Datasets/training_data.csv")
test_data = pd.read_csv("Datasets/test_data.csv")

X_train = train_data.drop(columns=["Action"])
y_train = train_data["Action"]
X_test = test_data.drop(columns=["Action"])
y_test = test_data["Action"]


param_combinations = [
    ((50,), "relu", "adam", 0.001),
    ((50,), "tanh", "adam", 0.001),
    ((100,), "relu", "adam", 0.001),
    ((100, 50), "relu", "adam", 0.001),
    ((150, 100), "relu", "adam", 0.001),
    ((100, 50), "tanh", "adam", 0.001),
]

print(f"Total parameter combinations: {len(param_combinations)}")

results = []

for i, (hidden, activation, solver, lr) in enumerate(param_combinations, start=1):
    print(f"\nTraining Model {i}/{len(param_combinations)}")
    print(f"Hidden={hidden}, Activation={activation}, Solver={solver}, LR={lr}")

    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation=activation,
        solver=solver,
        learning_rate_init=lr,
        early_stopping=True,
        max_iter=2500,  
        random_state=42
    )

    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        results.append({
            "Model_ID": i,
            "Hidden_Layers": hidden,
            "Activation": activation,
            "Solver": solver,
            "Learning_Rate_Init": lr,
            "Early_Stopping": True,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae
        })

    except Exception as e:
        print(f"Model {i} failed: {e}")
        results.append({
            "Model_ID": i,
            "Hidden_Layers": hidden,
            "Activation": activation,
            "Solver": solver,
            "Learning_Rate_Init": lr,
            "Early_Stopping": True,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)

print("\nTop Models (Lowest RMSE):")
print(results_df.head())

results_df.to_csv("mlp_results.csv", index=False)
print("\nAll results saved to 'mlp_results.csv'")

best_model_row = results_df.iloc[0]
print("\nBest Model Parameters:")
print(best_model_row)

best_params = {
    "hidden_layer_sizes": best_model_row["Hidden_Layers"],
    "activation": best_model_row["Activation"],
    "solver": best_model_row["Solver"],
    "learning_rate_init": best_model_row["Learning_Rate_Init"]
}

final_model = MLPRegressor(
    hidden_layer_sizes=best_params["hidden_layer_sizes"],
    activation=best_params["activation"],
    solver=best_params["solver"],
    learning_rate_init=best_params["learning_rate_init"],
    early_stopping=False,
    max_iter=2500,
    random_state=42
)

print("\nRetraining best model with early_stopping=False...")
final_model.fit(X_train, y_train)
y_pred_final = final_model.predict(X_test)

mse_final = mean_squared_error(y_test, y_pred_final)
rmse_final = np.sqrt(mse_final)
mae_final = mean_absolute_error(y_test, y_pred_final)

rmse_diff = abs(rmse_final - best_model_row["RMSE"])
mse_diff = abs(mse_final - best_model_row["MSE"])
mae_diff = abs(mae_final - best_model_row["MAE"])

print("\nRetrained Model (early_stopping=False) Results:")
print(f"MSE: {mse_final:.3f}")
print(f"RMSE: {rmse_final:.3f}")
print(f"MAE: {mae_final:.3f}")

print(f"\nDifference vs early_stopping=True → RMSE Δ: {rmse_diff:.3f}, MSE Δ: {mse_diff:.3f}, MAE Δ: {mae_diff:.3f}")

final_result_row = {
    "Model_ID": "BEST_MODEL_RETRAINED",
    "Hidden_Layers": best_params["hidden_layer_sizes"],
    "Activation": best_params["activation"],
    "Solver": best_params["solver"],
    "Learning_Rate_Init": best_params["learning_rate_init"],
    "Early_Stopping": False,
    "MSE": mse_final,
    "RMSE": rmse_final,
    "MAE": mae_final
}

results_df = pd.concat([results_df, pd.DataFrame([final_result_row])], ignore_index=True)
results_df.to_csv("mlp_results.csv", index=False)
print("\nUpdated results saved to 'mlp_results.csv'")
