from stable_baselines3 import PPO
from env.navigation_env import NavigationEnv

env = NavigationEnv(render=True)

model = PPO.load("nav_model")

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)

    if done or truncated:
        obs, _ = env.reset()