from stable_baselines3 import PPO
from env.navigation_env import NavigationEnv
import time

env = NavigationEnv(render=True)

model = PPO.load("nav_model")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, truncated, info = env.step(action)

    time.sleep(0.001)

    if done or truncated:
        obs, info = env.reset()