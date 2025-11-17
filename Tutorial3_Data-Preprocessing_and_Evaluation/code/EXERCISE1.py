import gymnasium as gym
from map_loader import prepare_for_env
from pynput import keyboard
import csv
import numpy as np
import os
import time


# Output folder and CSV paths

OUTPUT_DIR = "DATA_T3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_ADDITIONAL = os.path.join(OUTPUT_DIR, "dataset_additional.csv")
CSV_ORIGINAL = os.path.join(OUTPUT_DIR, "dataset_original.csv")


# Create CSV headers

with open(CSV_ADDITIONAL, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "action",
        "passenger_in_taxi",
        "dist_to_passenger",
        "dist_to_destination",
        "rel_passenger_row",
        "rel_passenger_col",
        "rel_dest_row",
        "rel_dest_col"
    ])

with open(CSV_ORIGINAL, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "action",
        "taxi_row",
        "taxi_col",
        "passenger_index",
        "destination_index",
        "reward"
    ])


# Key bindings

key_action = {
    keyboard.Key.down: 0,  # move south
    keyboard.Key.up: 1,    # move north
    keyboard.Key.right: 2, # move east
    keyboard.Key.left: 3,  # move west
    keyboard.KeyCode.from_char('p'): 4, # pick up passenger
    keyboard.Key.enter: 5  # drop off passenger
}

current_action = None

def on_press(key):
    global current_action
    if key in key_action:
        current_action = key_action[key]

def on_release(key):
    # We keep current_action until processed; do nothing here
    pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


# Feature extraction functions

def get_additional_features_from_state(state, env):
    taxi_row, taxi_col, pass_idx, dest_idx = env.unwrapped.decode(state)
    locs = env.unwrapped.locs

    if pass_idx < len(locs):
        pass_row, pass_col = locs[pass_idx]
    else:
        pass_row, pass_col = taxi_row, taxi_col

    dest_row, dest_col = locs[dest_idx]

    passenger_in_taxi = int(pass_idx >= len(locs))
    dist_to_passenger = abs(taxi_row - pass_row) + abs(taxi_col - pass_col)
    dist_to_destination = abs(taxi_row - dest_row) + abs(taxi_col - dest_col)
    rel_passenger_row = pass_row - taxi_row
    rel_passenger_col = pass_col - taxi_col
    rel_dest_row = dest_row - taxi_row
    rel_dest_col = dest_col - taxi_col

    return [
        passenger_in_taxi,
        dist_to_passenger,
        dist_to_destination,
        rel_passenger_row,
        rel_passenger_col,
        rel_dest_row,
        rel_dest_col
    ]

def get_original_features_from_state(state, env, last_reward):
    taxi_row, taxi_col, pass_idx, dest_idx = env.unwrapped.decode(state)
    return [taxi_row, taxi_col, pass_idx, dest_idx, last_reward]


# Map lists

train_maps = [f"map_{i}.txt" for i in range(1, 9)]
test_maps  = [f"map_{i}.txt" for i in range(9, 11)]
maps_to_collect = train_maps + test_maps


# Main loop

EPISODES_PER_MAP = 10

print("Controls: arrow keys = move, 'p' = pickup, Enter = dropoff.")
time.sleep(0.5)

for map_file in maps_to_collect:
    print(f"\n=== Collecting data for {map_file} ===")
    env = gym.make("Taxi-v3", desc=prepare_for_env(map_file), render_mode="human")

    for ep in range(EPISODES_PER_MAP):
        print(f"Episode {ep+1}/{EPISODES_PER_MAP} - play now")
        observation, info = env.reset(seed=None)
        terminated, truncated = False, False
        last_reward = 0

        while not (terminated or truncated):
            if current_action is None:
                env.render()
                time.sleep(0.05)
                continue

            action_to_take = current_action
            current_action = None

            next_obs, reward, terminated, truncated, info = env.step(action_to_take)

            # Features before action
            add_feats = get_additional_features_from_state(observation, env)
            orig_feats = get_original_features_from_state(observation, env, reward)

            # Append to CSVs
            with open(CSV_ADDITIONAL, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([action_to_take] + add_feats)

            with open(CSV_ORIGINAL, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([action_to_take] + orig_feats)

            observation = next_obs
            last_reward = reward
            env.render()

        print("End of episode.")
    env.close()

print("\n✅ Data collection finished.")
print(f" - Additional features saved to: {CSV_ADDITIONAL}")
print(f" - Original features saved to:   {CSV_ORIGINAL}")
