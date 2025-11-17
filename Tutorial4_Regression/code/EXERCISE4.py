import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

train_data = pd.read_csv("Datasets/training_data.csv")
test_data = pd.read_csv("Datasets/test_data.csv")

X_train = train_data.drop(columns=["Action"])
y_train = train_data["Action"]
X_test = test_data.drop(columns=["Action"])
y_test = test_data["Action"]


# Decision Tree (best parameters)
dt_model = DecisionTreeRegressor(
    criterion='squared_error',
    splitter='best',
    max_depth=5,
    min_samples_split=2,
    max_features=None,
    random_state=42
)
dt_model.fit(X_train, y_train)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# MLP Regressor
mlp_model = MLPRegressor(
    hidden_layer_sizes=(100,50),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=2500,
    early_stopping=True,
    random_state=42
)
mlp_model.fit(X_train, y_train)

def evaluate_agent(env, model, episodes=3, render=True):
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if render:
                env.render()

            # Convert obs to DataFrame with column names to avoid MLP warning
            obs_array = np.array(obs).reshape(1, -1)
            obs_df = pd.DataFrame(obs_array, columns=X_train.columns)

            action = model.predict(obs_df)[0]
            action = np.clip(action, -2.0, 2.0)  # Pendulum bounds

            obs, reward, terminated, truncated, _ = env.step([action])
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}")

    avg_reward = np.mean(rewards)
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward:.2f}\n")
    return avg_reward

env = gym.make("Pendulum-v1", render_mode="human")  

print("\nEvaluating Decision Tree Agent...")
reward_dt = evaluate_agent(env, dt_model, episodes=3, render=True)

print("\nEvaluating Linear Regression Agent...")
reward_lr = evaluate_agent(env, lr_model, episodes=3, render=True)

print("\nEvaluating MLP Agent...")
reward_mlp = evaluate_agent(env, mlp_model, episodes=3, render=True)

env.close()

results = pd.DataFrame([
    {"Model": "Decision Tree", "Avg Reward": reward_dt},
    {"Model": "Linear Regression", "Avg Reward": reward_lr},
    {"Model": "MLP Regressor", "Avg Reward": reward_mlp}
]).sort_values(by="Avg Reward", ascending=False)

print("\nAgent Performance Comparison:")
print(results)

results.to_csv("agent_comparison_results.csv", index=False)
print("\nResults saved to 'agent_comparison_results.csv'")

