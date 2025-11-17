import numpy as np
import gymnasium as gym
from map_loader import prepare_for_env

class SimpleTaxiAgent:

    def __init__(self, env):
        self.env = env.unwrapped
        self.passenger_loc = 2      # Fixed pickup location index
        self.destination_loc = 3    # Fixed dropoff location index
        self.last_position = None
        self.stuck_count = 0
        self.alternative_action_index = 0

    def decode_state(self, state):
        out = state
        dest = out % 4
        out //= 4
        passenger_loc = out % 5
        out //= 5
        taxi_col = out % 5
        out //= 5
        taxi_row = out
        return taxi_row, taxi_col, passenger_loc, dest

    def get_fixed_positions(self):
        return [(0, 0), (0, 4), (4, 0), (4, 3)]

    def select_action(self, state):
        taxi_row, taxi_col, passenger_loc, dest = self.decode_state(state)
        fixed_positions = self.get_fixed_positions()

        # Check if stuck (same position as last time)
        current_pos = (taxi_row, taxi_col)
        if self.last_position == current_pos:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            self.alternative_action_index = 0
        self.last_position = current_pos

        # Determine target based on whether passenger is in taxi
        if passenger_loc != 4:  # Passenger not in taxi
            target_row, target_col = fixed_positions[self.passenger_loc]
            # Check if at pickup location
            if (taxi_row, taxi_col) == (target_row, target_col):
                self.stuck_count = 0
                return 4  # Pickup
        else:  # Passenger in taxi
            target_row, target_col = fixed_positions[self.destination_loc]
            # Check if at dropoff location
            if (taxi_row, taxi_col) == (target_row, target_col):
                self.stuck_count = 0
                return 5  # Dropoff

        # If stuck, try alternative moves by cycling through all actions
        if self.stuck_count > 0:
            actions = [0, 1, 2, 3]  # south, north, east, west
            action = actions[self.alternative_action_index % 4]
            self.alternative_action_index += 1
            return action

        # Navigate to target (normal pathfinding)
        row_diff = target_row - taxi_row
        col_diff = target_col - taxi_col

        # Prioritize the larger difference
        if abs(row_diff) >= abs(col_diff):
            if row_diff > 0:
                return 0  # south
            elif row_diff < 0:
                return 1  # north
            elif col_diff > 0:
                return 2  # east
            elif col_diff < 0:
                return 3  # west
        else:
            if col_diff > 0:
                return 2  # east
            elif col_diff < 0:
                return 3  # west
            elif row_diff > 0:
                return 0  # south
            elif row_diff < 0:
                return 1  # north

        return 0  # default

if __name__ == "__main__":
    env = gym.make("Taxi-v3", desc=prepare_for_env("map_1.txt"))
    env.unwrapped.desc = np.asarray(env.unwrapped.desc)
    state = env.reset(seed=42)

    if isinstance(state, tuple):
        state = list(state)
    if isinstance(state, list) and isinstance(state[0], int):
        state = state[0]

    agent = SimpleTaxiAgent(env)

    print(f"Initial state: {state}")
    print(f"Target pickup: {agent.get_fixed_positions()[agent.passenger_loc]}")
    print(f"Target destination: {agent.get_fixed_positions()[agent.destination_loc]}")

    for step in range(100):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        if isinstance(next_state, tuple):
            next_state = list(next_state)
        if isinstance(next_state, list) and isinstance(next_state[0], int):
            next_state = next_state[0]

        taxi_row, taxi_col, passenger_loc, dest = agent.decode_state(next_state)
        print(f"Step {step+1}: Action={action}, Taxi=({taxi_row},{taxi_col}), "
              f"Passenger={passenger_loc}, Destination={dest}")

        if action == 4:
            print("PASSENGER PICKED")
        elif action == 5:
            print("PASSENGER DROPPED OFF AT DESTINATION")

        state = next_state
        if terminated or truncated:
            print(f"EPISODE FINISHED")
            break

    env.close()