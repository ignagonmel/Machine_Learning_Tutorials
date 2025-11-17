import os
import ast
import gymnasium as gym
import pandas as pd
import numpy as np
from gymnasium.envs.toy_text.taxi import TaxiEnv
from sklearn.tree import DecisionTreeClassifier
from map_loader import prepare_for_env


# 1. Load top 3 models

TOP3_FILE = "RESULTS_T3/q2_top3_global.csv"
DATA_PATH = "DATA_T3"

top3_df = pd.read_csv(TOP3_FILE)
print("✅ Loaded top 3 models from previous exercise\n")
print(top3_df, "\n")

# Convert stringified dicts into real Python dicts
top3_df["params"] = top3_df["params"].apply(ast.literal_eval)


# 2. Load datasets used for training

datasets = {
    "additional": pd.read_csv(os.path.join(DATA_PATH, "dataset_additional.csv")),
    "mi": pd.read_csv(os.path.join(DATA_PATH, "dataset_mi.csv")),
    "rfe": pd.read_csv(os.path.join(DATA_PATH, "dataset_rfe.csv"))
}


# 3. Feature extraction function


def get_engineered_features_from_state(state, env):
    """Extract features for ADDITIONAL, MI, RFE datasets."""
    taxi_row, taxi_col, pass_idx, dest_idx = env.unwrapped.decode(state)
    locs = env.unwrapped.locs

    # Determine passenger's location
    if pass_idx < len(locs):
        pass_row, pass_col = locs[pass_idx]
    else:
        pass_row, pass_col = taxi_row, taxi_col # Passenger in taxi

    dest_row, dest_col = locs[dest_idx]

    passenger_in_taxi = int(pass_idx >= len(locs))
    dist_to_passenger = abs(taxi_row - pass_row) + abs(taxi_col - pass_col)
    dist_to_destination = abs(taxi_row - dest_row) + abs(taxi_col - dest_col)
    rel_passenger_row = pass_row - taxi_row
    rel_passenger_col = pass_col - taxi_col
    rel_dest_row = dest_row - taxi_row
    rel_dest_col = dest_col - taxi_col

    return np.array([
        passenger_in_taxi,
        dist_to_passenger,
        dist_to_destination,
        rel_passenger_row,
        rel_passenger_col,
        rel_dest_row,
        rel_dest_col
    ], dtype=np.float32)


def run_model_in_env(clf, env, feature_names, render, max_steps=100):
    """Run a trained DecisionTree model inside the Taxi environment."""
    obs, info = env.reset()
    terminated, truncated = False, False
    steps = 0
    success = False

    while not (terminated or truncated) and steps < max_steps:
        features = get_engineered_features_from_state(obs, env)
        

        num_features_needed = len(feature_names)
        features_subset = features[:num_features_needed]
        
        if len(features_subset) != num_features_needed:
             print(f"Error: Feature count mismatch. Expected {num_features_needed}, got {len(features_subset)}. Skipping step.")
             break


        # Create DataFrame to match feature columns expected by model
        X = pd.DataFrame([features_subset], columns=feature_names)

        # Predict the action
        action = clf.predict(X)[0]
        
        # Take the step
        obs, reward, terminated, truncated, info = env.step(action)

        if render:
            env.render()

        # Check if successful completion
        if terminated and reward == 20:
            success = True
            break

        steps += 1

    return success, steps


# 4. Load maps 

train_maps = [f"map_{i}.txt" for i in range(1, 9)]
test_maps = [f"map_{i}.txt" for i in range(9, 11)]


# 5. Deploy models on training maps 

print("================= QUESTION 1 =================")
print("Evaluating the 3 best models (Engineered Features) on TRAIN maps...\n")

train_results = []

for idx, row in top3_df.iterrows():
    dataset_name = row["Dataset"]
    params = row["params"]

    print(f"\n🚕 Testing model {idx+1} ({dataset_name.upper()}) on TRAIN maps...")
    
    df = datasets[dataset_name]
    feature_names = [c for c in df.columns if c != "action"]
    
    X = df[feature_names]
    y = df["action"]

    # Train model again (no CV, full dataset)
    clf = DecisionTreeClassifier(**params, random_state=42)
    clf.fit(X, y)

    total_maps = len(train_maps)
    successes = 0
    total_steps_success = 0 

    for map_file in train_maps:
        env = gym.make("Taxi-v3", desc=prepare_for_env(map_file))
        success, steps = run_model_in_env(clf, env, X.columns.tolist(), render=False) 
        
        if success:
            successes += 1
            total_steps_success += steps 
        
        print(f"Map {map_file}: {'SUCCESS' if success else 'FAIL'} ({steps} steps)")
        env.close()

    avg_steps = total_steps_success / successes if successes > 0 else 0 
    train_results.append({
        "Model": f"Model_{idx+1}_{dataset_name}",
        "Train Successes": successes,
        "Train Total Maps": total_maps,
        "Train Success Rate": successes / total_maps,
        "Avg Steps (Success)": avg_steps
    })

train_df = pd.DataFrame(train_results)
print("\n===== TRAIN RESULTS =====")
print(train_df.round(3))


# 6. Deploy models on test maps

print("\n================= QUESTION 2 =================")
print("Evaluating the same 3 models (Engineered Features) on TEST maps...\n")

test_results = []

for idx, row in top3_df.iterrows():
    dataset_name = row["Dataset"]
    params = row["params"]

    print(f"\n🚕 Testing model {idx+1} ({dataset_name.upper()}) on TEST maps...")

    df = datasets[dataset_name]
    feature_names = [c for c in df.columns if c != "action"]
    
    X = df[feature_names]
    y = df["action"]

    clf = DecisionTreeClassifier(**params, random_state=42)
    clf.fit(X, y)

    total_maps = len(test_maps)
    successes = 0
    total_steps_success = 0 

    for map_file in test_maps:
        env = gym.make("Taxi-v3", desc=prepare_for_env(map_file))
        success, steps = run_model_in_env(clf, env, X.columns.tolist(), render=False)
        
        if success:
            successes += 1
            total_steps_success += steps
        
        print(f"Map {map_file}: {'SUCCESS' if success else 'FAIL'} ({steps} steps)")
        env.close()

    avg_steps = total_steps_success / successes if successes > 0 else 0
    test_results.append({
        "Model": f"Model_{idx+1}_{dataset_name}",
        "Test Successes": successes,
        "Test Total Maps": total_maps,
        "Test Success Rate": successes / total_maps,
        "Avg Steps (Success)": avg_steps
    })

test_df = pd.DataFrame(test_results)
print("\n===== TEST RESULTS =====")
print(test_df.round(3))


# 7. Save results (No change)

os.makedirs("RESULTS_T3", exist_ok=True)
train_df.to_csv("RESULTS_T3/q3_train_deploy_results.csv", index=False)
test_df.to_csv("RESULTS_T3/q3_test_deploy_results.csv", index=False)
print("\n✅ Saved deployment results in 'RESULTS_T3/' folder.")