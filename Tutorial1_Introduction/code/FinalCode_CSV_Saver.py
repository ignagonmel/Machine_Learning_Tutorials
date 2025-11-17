import gymnasium as gym
from map_loader import prepare_for_env
import pandas as pd
env = gym . make ("Taxi-v3", desc = prepare_for_env ("map_1.txt") , render_mode ="human")
observation , info = env . reset ( seed =42)
experiences = [] #LISTA PARA ALMACENAR LAS EXPERIENCIAS
for _ in range (1000) :
    action = env . action_space . sample ()
    observation , reward , terminated , truncated , info = env . step ( action )
    experiences.append({
        "state": observation,
        "action": action,
        "reward": reward,
        "terminated": terminated
    })
if terminated or truncated :
    observation , info = env . reset ()

env . close ()
pd.DataFrame(experiences).to_csv("experiences_auto.csv", index=False)
print("Experiencias guardadas en experiences_auto.csv")