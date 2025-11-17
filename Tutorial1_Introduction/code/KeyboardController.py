import gymnasium as gym
from map_loader import prepare_for_env
from pynput import keyboard
import pandas as pd
import time

#INICIALIZAR ENTORNO
env = gym.make("Taxi-v3", desc=prepare_for_env("map_1.txt"), render_mode="human")
obs, info = env.reset(seed=42)

# DICCIONARIO DE ACCIONES
key_to_action = {
    'w': 1,  # NORTE
    's': 0,  # SUR
    'a': 3,  # IZQ
    'd': 2,  # DCHA
    'p': 4,  # RECOGER
    'o': 5   # DEJAR
}

experiences = [] #LISTA PARA ALMACENAR LAS EXPERIENCIAS
current_action = None

def on_press(key):
    global current_action
    try:
        if key.char in key_to_action:
            current_action = key_to_action[key.char]
    except AttributeError:
        pass 

listener = keyboard.Listener(on_press=on_press)
listener.start()

print("MOVEMENTS: w = up, s = down a = left d = right p = pick-up o = drop-off")

for _ in range(1000): #MAX 1000 experiences

    while current_action is None:
        time.sleep(0.05)

    action = current_action
    current_action = None #EL OBJETIVO DE ESTA LINEA ES QUE LA ACCIÓN CAPTADA NO SE REPITA DE FORMA INFINITA

    next_obs, reward, terminated, truncated, info = env.step(action)

    # GUARDAMOS EN LA LISTA DE EXPERIENCIAS LA ÚLTIMA EN FORMA DE DICCIONARIO
    experiences.append({
        "state": obs,
        "action": action,
        "reward": reward,
        "next_state": next_obs,
        "terminated": terminated
    })

    obs = next_obs

    if terminated or truncated:
        obs, info = env.reset()

env.close()

# GUARDAMOS LA LISTA "EXPERIENCES" EN UN CSV
pd.DataFrame(experiences).to_csv("experiences.csv", index=False)
print("Experiencias guardadas en experiences.csv")
