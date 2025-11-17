import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import gymnasium as gym
from map_loader import prepare_for_env
from pynput import keyboard
import time
import numpy as np



df = pd.read_csv("experiences.csv")


X = df[["state"]]     # input: game state
y = df["action"]      # output: action to predict


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


clf = DecisionTreeClassifier(criterion= 'gini', splitter= 'best', max_depth= 10, min_samples_split= 2, class_weight= 'balanced', random_state=42)
clf.fit(X_train, y_train) # Train tree


env = gym . make ("Taxi-v3", desc = prepare_for_env ("map_1.txt") , render_mode ="human")
observation , info = env . reset ( seed =42)
while (1):
    state_input = pd.DataFrame({"state": [observation]})  
    action = int(clf.predict(state_input)[0])     

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env . close ()
